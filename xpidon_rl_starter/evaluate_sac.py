"""Evaluate a saved SAC policy vs random-action and no-change policies.
All evaluations are XPIDON predictions, not FE/material certification.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
from predictor import PersistentXPIDON,PARAMS
from rl_env import CureDesignEnv


def get_baseline(path,case_id):
    with open(path,newline='') as f:
        items=[r for r in csv.DictReader(f) if r['case_id']==case_id]
    if len(items)!=1:raise ValueError('Baseline not found/unique: '+case_id)
    return {k:(v if k in ('case_id','study','label','curve_label') else float(v)) for k,v in items[0].items()}


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--baseline',default='cure_cycle_1')
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('rl_evaluation'))
    p.add_argument('--episodes',type=int,default=20)
    p.add_argument('--seed',type=int,default=20261020)
    p.add_argument('--alpha-req',type=float,default=None)
    p.add_argument('--t-limit',type=float,default=None)
    p.add_argument('--run-info',type=Path)
    p.add_argument('--reward-mode',choices=['v2','v3'],default=None)
    p.add_argument('--reference-summary',type=Path,help='Optional existing baseline summary.json for predictor agreement')
    p.add_argument('--alpha-tol',type=float,default=0.002)
    p.add_argument('--temp-tol',type=float,default=0.5)
    return p.parse_args()


def main():
    a=parse_args()
    if a.episodes<1:raise ValueError('episodes must be positive')
    info=json.loads(a.run_info.read_text()) if a.run_info else {}
    alpha=float(a.alpha_req if a.alpha_req is not None else info.get('alpha_requirement',0.82))
    temp=float(a.t_limit if a.t_limit is not None else info.get('temperature_limit_C',195.))
    baseline=get_baseline(a.cases,a.baseline)
    predictor=PersistentXPIDON(a.manifest,baseline)
    env=CureDesignEnv(predictor,alpha_requirement=alpha,temperature_limit_C=temp,
                      max_steps=int(info.get('max_steps_per_episode',8)),
                      action_fraction=float(info.get('action_fraction',0.12)),
                      reward_mode=a.reward_mode or info.get('reward_mode','v2'),
                      action_decay=float(info.get('action_decay',0.0)))
    from stable_baselines3 import SAC
    model=SAC.load(str(a.model),device='cpu')
    if tuple(model.observation_space.shape)!=tuple(env.observation_space.shape) or tuple(model.action_space.shape)!=tuple(env.action_space.shape):
        raise ValueError('Saved SAC policy spaces do not match current environment')
    a.output.mkdir(parents=True,exist_ok=True)
    baseline_x=np.array([baseline[k] for k in PARAMS])
    baseline_metrics=predictor.evaluate(baseline_x)
    consistency={'baseline_recomputed':baseline_metrics,'status':'no_reference_supplied'}
    if a.reference_summary:
        ref=json.loads(a.reference_summary.read_text())
        keys={'alpha_min':'sampled_alpha_min_at_heating_hold_end','peak_T_C':'sampled_peak_part_temperature_C'}
        errors={k:abs(baseline_metrics[k]-float(ref[v])) for k,v in keys.items()}
        consistency.update(reference=str(a.reference_summary),absolute_errors=errors,
            status='PASS' if errors['alpha_min']<=a.alpha_tol and errors['peak_T_C']<=a.temp_tol else 'FAIL')
        if consistency['status']=='FAIL':
            (a.output/'consistency.json').write_text(json.dumps(consistency,indent=2)+'\n')
            raise RuntimeError('XPIDON predictor disagrees with supplied baseline; stop before evaluating policy')
    (a.output/'consistency.json').write_text(json.dumps(consistency,indent=2)+'\n')
    records=[]
    initial_rng=np.random.default_rng(a.seed)
    for episode in range(a.episodes):
        init=initial_rng.uniform(predictor.lo,predictor.hi)
        for mode in ('SAC','Random','NoChange'):
            obs,meta=env.reset(seed=a.seed+episode,options={'x':init})
            for step in range(env.max_steps):
                if mode=='SAC':action,_=model.predict(obs,deterministic=True)
                elif mode=='Random':action=initial_rng.uniform(-1,1,size=6).astype(np.float32)
                else:action=np.zeros(6,dtype=np.float32)
                obs,reward,done,trunc,inf=env.step(action)
                m=inf['metrics'];x=inf['x']
                records.append({'episode':episode,'method':mode,'step':step+1,'reward':float(reward),
                    'score':inf['score'],'feasible':bool(inf['feasible']),
                    'hold_end_min':m['hold_end_min'],'alpha_min':m['alpha_min'],
                    'peak_T_C':m['peak_T_C'],'alpha_spread':m['alpha_spread'],
                    **dict(zip(PARAMS,x))})
                if done or trunc:break
    with (a.output/'trajectories.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    final=[r for r in records if r['step']==env.max_steps]
    stats={}
    for method in ('SAC','Random','NoChange'):
        vals=[r for r in final if r['method']==method]
        feas=[r for r in vals if r['feasible']]
        stats[method]={'episodes':len(vals),'feasible_rate':len(feas)/len(vals),
            'mean_final_score':float(np.mean([r['score'] for r in vals])),
            'mean_final_alpha_min':float(np.mean([r['alpha_min'] for r in vals])),
            'best_feasible':min(feas,key=lambda z:z['hold_end_min']) if feas else None}
    result={'research_only':True,'thresholds':{'alpha_min':alpha,'peak_T_C':temp},
        'baseline_metrics':baseline_metrics,'predictor_consistency':consistency,
        'evaluation':stats,'unique_xpidon_evaluations':len(predictor.cache),
        'comparison_note':'Same initial states and number of actions; SAC training budget is additional and must be counted in fair end-to-end comparisons.'}
    (a.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,4))
        for mode in ('SAC','Random','NoChange'):
            arr=np.array([ [r['score'] for r in records if r['method']==mode and r['step']==s] for s in range(1,env.max_steps+1)])
            ax.plot(np.arange(1,env.max_steps+1),np.mean(arr,axis=1),label=mode)
        ax.set(xlabel='Design refinement step',ylabel='Mean score',title='Held-out start states: sequential design evaluation')
        ax.legend();fig.tight_layout();fig.savefig(a.output/'score_trajectories.png',dpi=160);plt.close(fig)
    except ImportError:pass
    print(json.dumps({'output':str(a.output),'evaluation':{k:{'feasible_rate':v['feasible_rate'],'mean_final_score':v['mean_final_score']} for k,v in stats.items()},'unique_predictions':len(predictor.cache)},indent=2))

if __name__=='__main__':main()
