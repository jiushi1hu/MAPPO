"""Paired epoch replay for snapshots produced by analysis.rebound_capture."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from analysis.rebound_capture import HISTORY, history, make_runner, restore_rng
from harl.algorithms.actors.shared_mappo import joint_shared_actor_batch
from harl.envs.cascade_reservoir.episodes import episode_environment


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def save_row(path, row):
    with path.open('a') as stream:
        print(json.dumps(row, allow_nan=False), file=stream)


def verify_batch(batch):
    fields = ['effective_action_index', 'effective_action_log_probability', 'raw_advantage',
              'own_value_frozen', 'learning_reward']
    cb = batch['critic']
    mask = np.zeros((1825, 6, 5), dtype=bool)
    errors = np.zeros(len(fields))
    with (HISTORY / 'training_seed1_trajectory.csv').open('rb') as f:
        header = f.readline().decode().strip().split(',')
        indices = [header.index(k) for k in fields]
        for line in f:
            prefix = line.split(b',', 2)
            if prefix[1] != b'46':
                if prefix[1] == b'47':
                    break
                continue
            row = line.split(b',')
            t, lane, j = int(row[6])-1, int(row[3]), int(row[9])
            assert not mask[t, lane, j]
            mask[t, lane, j] = True
            assert int(row[4]) == batch['windows'][lane]
            assert int(row[47]) == 1-batch['actors'][j].active_masks[t,lane,0]
            assert float(row[66]) == cb.masks[t+1,lane,0]
            buf = batch['actors'][j]
            recovered = [buf.actions[t,lane,0], buf.action_log_probs[t,lane,0],
                         batch['advantages'][t,lane,j],
                         cb.returns[t,lane,j]-batch['advantages'][t,lane,j],
                         cb.rewards[t,lane,j]]
            errors = np.maximum(errors, np.abs(np.array(recovered)-[float(row[k]) for k in indices]))
    assert mask.all() and np.all(errors < [1e-8, 1e-6, 1e-5, 2e-5, 1e-6]), errors
    return dict(rows=int(mask.sum()), maximum_errors=dict(zip(fields, errors.tolist())))


@torch.no_grad()
def probabilities(net, obs):
    vectors = torch.as_tensor(obs, dtype=torch.float32)
    return net.act.action_out(net.base(vectors)).probs.numpy()


def state_dict_distance(first, second):
    return sum((first[k]-second[k]).double().square().sum().item() for k in first)**0.5


def fork_epochs(runner, folder, output):
    batch = load(folder/'batch_46.pt')
    save_row(output/'verification.jsonl', verify_batch(batch))
    core = runner.actor[0].core
    start = load(folder/'policy_45.pt')['actors'][0]
    end = load(folder/'policy_46.pt')['actors'][0]
    advantages = [batch['advantages'][..., j:j+1].copy() for j in range(5)]
    joint, adv = joint_shared_actor_batch(core, batch['actors'], advantages, 5)
    assert core.actor_num_mini_batch == 1 and core.ppo_epoch == 5
    core.actor.load_state_dict(start)
    obs = joint.obs[:-1].reshape(-1,21)
    old = probabilities(core.actor, obs)
    core.ppo_epoch = 1  # Diagnostic driver calls the unchanged single-epoch optimizer five times.
    for arm, rate in [('original', .0003), ('half', .00015), ('quarter', .000075)]:
        core.actor.load_state_dict(start)
        core.actor_optimizer.load_state_dict(load(folder/'optim_45.pt')['actor'])
        for group in core.actor_optimizer.param_groups:
            group['lr'] = rate
        restore_rng(batch['rng'])
        for epoch in range(1,6):
            info = core.train(joint, adv, 'EP', diagnostics=True, advantages_normalized=True)
            weights = core.actor.state_dict()
            new = probabilities(core.actor, obs)
            actions = joint.actions.reshape(-1).astype(int)
            ratio = new[np.arange(len(new)),actions]/old[np.arange(len(old)),actions]
            grad = sum(p.grad.double().square().sum().item() for p in core.actor.parameters()
                       if p.grad is not None)**.5
            save_row(output/'optimizer.jsonl', dict(arm=arm,epoch=epoch,lr=rate,
                parameter_delta=state_dict_distance(weights,start), gradient_pre=float(info['actor_grad_norm']),
                gradient_post=grad, surrogate=info['clipped_surrogate_after'],
                kl=float((old*np.log(old/new)).sum(1).mean()),
                entropy=float(-(new*np.log(new)).sum(1).mean()),
                clip_fraction=float((abs(ratio-1)>.2).mean()),
                mode_turnover=float((old.argmax(1)!=new.argmax(1)).mean())))
            torch.save(dict(actor=weights, optimizer=core.actor_optimizer.state_dict()),
                       output/f'{arm}_{epoch}.pt')
        if arm == 'original':
            assert all(torch.equal(end[k],weights[k]) for k in end)
            save_row(output/'verification.jsonl',dict(original_epoch5_bitwise_matches_update46=True))


def measure(env, commands):
    env.step(commands)
    r = env.last_transition
    return dict(actions=commands.copy(), release=r['release'], storage=r['storage_end'],
                deficit=r['service_shortfall_integrals']['ecological_flow']/1e9,
                energy=r['energy']/1e6, mixed=r['mixed_rewards'], system=r['system_reward'],
                lo=r['low'], hi=r['high'])


def stack(rows):
    return {k: np.asarray([r[k] for r in rows]) for k in rows[0]}


def rollout(runner, net):
    years = runner.periodic_window['source_years']
    env = episode_environment(runner.data,runner.physical_config,years,1825,100)
    states, rows = [], []
    for _ in range(1825):
        states.append(copy.copy(env))
        prob = probabilities(net,env.actor_observations())
        rows.append(measure(env,prob.argmax(1)))
    return states,stack(rows)


def probe(state, original, alternate, net, horizon):
    """Change the initial command only; both continuations use the same new actor."""
    traces = []
    for first in (original, alternate):
        env = copy.copy(state)
        command = first
        rows = []
        for t in range(horizon):
            if t:
                command = probabilities(net, env.actor_observations()).argmax(1)
            rows.append(measure(env, command))
        traces.append(stack(rows))
    result = {}
    for h in (1, 7, 30):
        if h > horizon:
            continue
        old, new = traces
        result[h] = dict(
            deficit=(new['deficit'][:h]-old['deficit'][:h]).sum(0).tolist(),
            storage=((new['storage'][h-1]-old['storage'][h-1])/1e9).tolist(),
            energy=float((new['energy'][:h]-old['energy'][:h]).sum()),
            mixed=(.995**np.arange(h) @ (new['mixed'][:h]-old['mixed'][:h])).tolist())
    return result


def compare(runner, net, base_net, states, base, name, output):
    _, path = rollout(runner,net)
    obs = np.stack([e.actor_observations() for e in states])
    orig = probabilities(base_net,obs)
    prob = probabilities(net,obs)
    acts = prob.argmax(2)
    np.testing.assert_array_equal(orig.argmax(2),base['actions'])
    fixed = []
    isolated = np.zeros_like(base['deficit'])
    isolated_system = np.zeros_like(isolated)
    for i,e in enumerate(states):
        commands = acts[i]
        fixed.append(measure(copy.copy(e),commands))
        for j in np.flatnonzero(commands!=base['actions'][i]):
            a=base['actions'][i].copy()
            a[j]=commands[j]
            alt=measure(copy.copy(e),a)
            isolated[i,j]=alt['deficit'][j]-base['deficit'][i,j]
            isolated_system[i,j]=alt['deficit'].sum()-base['deficit'][i].sum()
    fixed=stack(fixed)
    q=runner.data.service_requirements.ecological_flow_m3s
    grid=base['lo'][...,None]+(base['hi']-base['lo'])[...,None]*np.arange(100)/99
    compliant=grid >= q[None,:,None]*(1-1e-10)
    mass_shift=((prob-orig)*compliant).sum(2)
    # This expectation holds each realized upstream release fixed.
    shortfall_grid=np.where(compliant,0,np.maximum(q[None,:,None]-grid,0))*86400/1e9
    expected_shortfall_shift=((prob-orig)*shortfall_grid).sum(2)
    sensitive = compliant.any(2) & ~compliant.all(2)
    baseline_complies = base['deficit'] == 0
    univariate_harm = isolated > 1e-12
    changes=path['deficit']-base['deficit']
    stor=path['storage']-base['storage']
    first=lambda flag: int(np.flatnonzero(flag)[0]+1) if np.any(flag) else None
    additional=(path['deficit']>0)&(base['deficit']==0)
    idx=first(np.any(abs(stor)>1,axis=1))
    row=dict(name=name,count=int((path['deficit']>0).sum()),deficit=float(path['deficit'].sum()),
        energy=float(path['energy'].sum()),system_return=float(.995**np.arange(1825)@path['system']),
        terminal_storage_delta=float(stor[-1].sum()/1e9),fixed_turnover=float((acts!=base['actions']).mean()),
        path_turnover=float((acts!=path['actions']).mean()),fixed_count_delta=int((fixed['deficit']>0).sum()-(base['deficit']>0).sum()),
        fixed_deficit_delta=float((fixed['deficit']-base['deficit']).sum()),
        isolated_deficit_delta=isolated.sum(0).tolist(),isolated_system_delta=isolated_system.sum(0).tolist(),
        mass_shift_on_sensitive=float(mass_shift[sensitive].mean()),
        mass_shift_by_reservoir=[float(mass_shift[:,j][sensitive[:,j]].mean()) for j in range(5)],
        harmfully_switched=int(univariate_harm.sum()),
        conditional_expected_deficit_delta_by_reservoir=expected_shortfall_shift.sum(0).tolist(),
        fixed_new_violations=int(((fixed['deficit']>0)&baseline_complies).sum()),
        fixed_relieved_violations=int(((fixed['deficit']==0)&~baseline_complies).sum()),
        first_command=first(np.any(acts!=base['actions'],axis=1)),first_storage=idx,
        first_new_violation=first(additional.any(1)),max_extra_day=int(changes.sum(1).argmax()+1),
        deficit_delta_checkpoints=[float(changes[:t].sum()) for t in (30,365,730,1095,1460,1825)],
        counts_by_reservoir=(path['deficit']>0).sum(0).tolist())
    if name in ('original_5','half_5','quarter_5'):
        for label,day in [('first',idx),('largest',row['max_extra_day'])]:
            if day is None:
                continue
            t=day-1
            old=base['actions'][t]
            new=acts[t]
            for j in np.flatnonzero(new!=old):
                command=old.copy()
                command[j]=new[j]
                save_row(output/'probes.jsonl',dict(name=name,site=label,day=day,
                    reservoir=int(j),old_bin=int(old[j]),new_bin=int(new[j]),
                    result=probe(states[t],old,command,net,min(30,1825-t))))
    save_row(output/'evaluations.jsonl',row)
    np.savez_compressed(output/f'{name}_daily.npz',**path,fixed_commands=acts,
        base_probability_old=np.take_along_axis(orig,base['actions'][...,None],axis=2)[...,0],
        base_probability_new=np.take_along_axis(orig,acts[...,None],axis=2)[...,0],
        new_probability_old=np.take_along_axis(prob,base['actions'][...,None],axis=2)[...,0],
        new_probability_new=np.take_along_axis(prob,acts[...,None],axis=2)[...,0],
        base_commands=base['actions'],base_storage=base['storage'],base_deficit=base['deficit'],
        fixed_deficit=fixed['deficit'],isolated_deficit_delta=isolated,isolated_system_delta=isolated_system,
        compliance_probability_delta=mass_shift)
    return row


def decompose(runner, output):
    """Compare identical commands at baseline and branch inventory on each date."""
    with (output/'path_decomposition.jsonl').open('x') as stream:
        for name in ('original_5', 'half_5', 'quarter_5'):
            env = episode_environment(runner.data, runner.physical_config,
                                      runner.periodic_window['source_years'], 1825, 100)
            with np.load(output/f'{name}_daily.npz') as trace:
                base = trace['base_deficit']
                fixed = trace['fixed_deficit']
                actual = trace['deficit']
                hybrid = []
                for i, commands in enumerate(trace['actions']):
                    hybrid.append(measure(copy.copy(env), trace['fixed_commands'][i])['deficit'])
                    observed = measure(env, commands)
                    np.testing.assert_array_equal(observed['deficit'], actual[i])
                hybrid = np.array(hybrid)
                cases = (base, fixed, hybrid, actual)
                legs = {}
                for label, before, after in zip(
                    ('policy_at_old_storage', 'changed_storage_same_commands',
                     'command_response_at_changed_storage'), cases, cases[1:]):
                    legs[label] = dict(deficit_by_reservoir=(after-before).sum(0).tolist(),
                                       count=int((after>0).sum()-(before>0).sum()))
                print(json.dumps(dict(name=name, legs=legs)), file=stream, flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--decompose-only', action='store_true')
    args=parser.parse_args()
    folder=args.folder
    records=history()
    runner=make_runner(records)
    expected=[x['window_start_years'] for x in records
              if x['record_type']=='training_update' and x['seed']==1]
    sampler=copy.deepcopy(runner.training_sampler)
    assert [list(sampler.next_batch()) for _ in expected]==expected
    output=folder/'epoch_branches'
    if args.decompose_only:
        decompose(runner,output)
        return
    output.mkdir(exist_ok=False)
    save_row(output/'verification.jsonl',dict(full_historical_sampling_plan_matches=len(expected)))
    fork_epochs(runner,folder,output)
    net=runner.actor[0].actor
    prev=copy.deepcopy(net)
    for update in range(40,47):
        net.load_state_dict(load(folder/f'policy_{update}.pt')['actors'][0])
        if update>40:
            compare(runner,net,prev,states,base,f'onset_{update}',output)
        prev.load_state_dict(net.state_dict())
        states,base=rollout(runner,prev)
        if update==45:
            reference,baseline=states,base
            initial=copy.deepcopy(prev)
            np.savez_compressed(output/'epoch_zero.npz',**base)
    for arm in ('original','half','quarter'):
        for epoch in range(1,6):
            name=f'{arm}_{epoch}'
            net.load_state_dict(load(output/f'{name}.pt')['actor'])
            row=compare(runner,net,initial,reference,baseline,name,output)
            print(name,row['count'],row['deficit'],flush=True)

    decompose(runner,output)


if __name__=='__main__':
    main()
