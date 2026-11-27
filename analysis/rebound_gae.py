"""Frozen actor-only GAE ablation for the saved seed-1 rebound batches."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analysis.rebound_branches import (history, joint_shared_actor_batch, load, make_runner,
    measure, probabilities, restore_rng, rollout, save_row, state_dict_distance)


def gradient_scale(core, joint, advantages, target):
    """Scale the policy term to match the reference pre-clip norm, keeping entropy fixed."""
    state = torch.get_rng_state()
    sample = next(joint.feed_forward_generator_actor(advantages, 1))
    torch.set_rng_state(state)
    obs, act, _, active, old, adv, available = sample
    logp, entropy, _ = core.evaluate_actions(obs, act, available, return_entropy_per_sample=True)
    ratio = (logp-torch.as_tensor(old)).exp()
    a = torch.as_tensor(adv)
    w = torch.as_tensor(active)
    loss = -(w*torch.minimum(ratio*a, ratio.clamp(.8,1.2)*a)).mean()
    ent_loss = -core.entropy_coef*(w*entropy.reshape(-1,1)).mean()
    params = tuple(core.actor.parameters())
    grad = torch.cat([v.reshape(-1) for v in torch.autograd.grad(loss,params,retain_graph=True)]).double()
    ent = torch.cat([v.reshape(-1) for v in torch.autograd.grad(ent_loss,params)]).double()
    aa, ab, cc = float(grad@grad), float(grad@ent), float(ent@ent)-target**2
    return (-ab+np.sqrt(ab**2-aa*cc))/aa


def run(folder, update, gradient_control=False):
    out = folder / f'gae_ablation_{update}'
    if gradient_control:
        with (out/'epochs.jsonl').open() as stream:
            targets = [r['grad'] for r in map(json.loads,stream) if r['arm']=='0.95_raw']
        assert len(targets)==5
        out=out/'gradient_control'
    out.mkdir(exist_ok=False)
    runner = make_runner(history())
    batch = load(folder / f'batch_{update}.pt')
    before = load(folder / f'policy_{update-1}.pt')
    after = load(folder / f'policy_{update}.pt')
    core = runner.actor[0].core
    core.actor.load_state_dict(before['actors'][0])
    runner.value_normalizer.load_state_dict(before['value_normalizer'])
    states, base = rollout(runner, core.actor)
    np.savez_compressed(out/'baseline.npz', **base)
    obs = np.stack([s.actor_observations() for s in states])
    old_prob = probabilities(core.actor, obs)
    old47 = base['actions'] == 47
    assert np.array_equal(old_prob.argmax(2), base['actions'])
    cb = batch['critic']
    raw = batch['advantages']
    rms = lambda x: np.sqrt(np.mean(x.astype(float)**2, axis=(0,1)))
    assert all(np.all(a.active_masks[:-1]) for a in batch['actors'])
    core.ppo_epoch = 1
    for lam in ((.98,.99) if gradient_control else (.95,.98,.99)):
        cb.gae_lambda = lam
        cb.compute_returns(cb.value_preds[-1].copy(),runner.value_normalizer)
        advantage = cb.returns[:-1]-runner.value_normalizer.denormalize(cb.value_preds[:-1])
        if lam == .95:
            np.testing.assert_array_equal(raw,advantage)
        for matched in ([True] if gradient_control else ([False] if lam==.95 else [False,True])):
            scale = rms(raw)/rms(advantage) if matched else np.ones(5)
            used = (advantage*scale).astype(np.float32)
            joint, pooled = joint_shared_actor_batch(core,batch['actors'],
                [used[...,i:i+1].copy() for i in range(5)],5)
            core.actor.load_state_dict(before['actors'][0])
            core.actor_optimizer.load_state_dict(load(folder/f'optim_{update-1}.pt')['actor'])
            restore_rng(batch['rng'])
            name=f'{lam:.2f}_'+('matched' if matched else 'raw')
            for epoch in range(1,6):
                multiplier = gradient_scale(core,joint,pooled,targets[epoch-1]) if gradient_control else 1.
                info=core.train(joint,pooled*multiplier,'EP',diagnostics=True,advantages_normalized=True)
                if gradient_control:
                    np.testing.assert_allclose(float(info['actor_grad_norm']),targets[epoch-1],rtol=1e-5)
                prob=probabilities(core.actor,obs)
                _, path=rollout(runner,core.actor)
                shift=(prob[...,82]-prob[...,47])-(old_prob[...,82]-old_prob[...,47])
                delta_storage=(path['storage']-base['storage']).sum(1)/1e9
                key=f'{name}_{epoch}'
                np.savez_compressed(out/f'{key}.npz', **path, fixed_modes=prob.argmax(2),
                                    margin_shift=shift)
                row=dict(arm=name, epoch=epoch, count=int((path['deficit']>0).sum()),
                    deficit=float(path['deficit'].sum()),energy=float(path['energy'].sum()),
                    storage=float(delta_storage[-1]), min_storage_delta=float(delta_storage.min()),
                    switches=int(((prob.argmax(2)==82)&old47).sum()),
                    margin_pp=float(100*shift[old47].mean()),
                    grad=float(info['actor_grad_norm']), raw_rms=rms(advantage).tolist(),
                    used_rms=rms(used).tolist(), scale=scale.tolist(), epoch_multiplier=float(multiplier),
                    step=state_dict_distance(core.actor.state_dict(),before['actors'][0]))
                if lam==.95 and epoch==5:
                    assert all(torch.equal(w,after['actors'][0][k]) for k,w in core.actor.state_dict().items())
                    row['matches_historical_weights']=True
                save_row(out/'epochs.jsonl',row)
            print(update,name,row['count'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('update',type=int,choices=(42,46))
    parser.add_argument('--gradient-control',action='store_true')
    args=parser.parse_args()
    run(args.folder,args.update,args.gradient_control)
