"""Terminal paired local-command contrasts on historical 47-to-82 switch states."""
from pathlib import Path
import argparse
import copy
import numpy as np
import torch
from analysis.rebound_branches import history, make_runner, load, probabilities, measure, rollout, save_row


@torch.no_grad()
def run(folder, update):
    out=folder/f'gae_ablation_{update}'
    r=make_runner(history())
    pre=load(folder/f'policy_{update-1}.pt')
    net=r.actor[0].actor
    net.load_state_dict(pre['actors'][0])
    r.critic.critic.load_state_dict(pre['critic'])
    r.value_normalizer.load_state_dict(pre['value_normalizer'])
    states,base=rollout(r,net)
    obs=np.stack([s.actor_observations() for s in states])
    post=copy.deepcopy(net)
    post.load_state_dict(load(folder/f'policy_{update}.pt')['actors'][0])
    switched=(probabilities(post,obs).argmax(2)==82)&(base['actions']==47)
    path=out/'terminal_probes.jsonl'
    path.touch(exist_ok=False)
    for j in range(5):
        visits=np.flatnonzero(switched[:,j] & (np.arange(1825)<=1705))
        if not len(visits):continue
        # Select an early and late quartile visit among eligible switches, with >=120 days remaining.
        for day in np.unique(visits[(np.array([.25,.75])*(len(visits)-1)).astype(int)]):
            envs=[copy.copy(states[day]) for _ in range(2)]
            diff=[]; values=[]
            for k in range(day,1825):
                o=np.stack([e.actor_observations() for e in envs])
                v=r.value_normalizer.denormalize(r.critic.critic(o.reshape(2,-1)).numpy())[:,j]
                values.append(v[1]-v[0])
                acts=probabilities(net,o).argmax(2)
                if k==day:
                    acts[:]=base['actions'][day]
                    acts[1,j]=82
                results=[measure(e,a) for e,a in zip(envs,acts)]
                metrics=np.array([[z['mixed'][j],z['system'],z['deficit'].sum(),z['energy'].sum(),z['storage'].sum()/1e9] for z in results])
                diff.append(metrics[1]-metrics[0])
            a=np.array(diff);v=np.r_[values,0.];discount=.995**np.arange(len(a))
            td=a[:,0]+.995*v[1:]-v[:-1]
            save_row(path,dict(reservoir=j,day=int(day+1),horizon=len(a),
                own=float(discount@a[:,0]),system=float(discount@a[:,1]),deficit=float(a[:,2].sum()),
                energy=float(a[:,3].sum()),terminal_storage=float(a[-1,4]),
                immediate=a[0].tolist(),bootstrap1=float(td[0]),
                gae=[float((discount*l**np.arange(len(a)))@td) for l in (.95,.98,.99)]))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path);parser.add_argument('update',type=int)
    args=parser.parse_args();run(args.folder,args.update)
