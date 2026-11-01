"""SAC train with reproducible seeds, periodic validation and diagnostics."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
from predictor import PersistentXPIDON, PARAMS
from rl_env import CureDesignEnv
from evaluate_sac import get_baseline


def evaluate_policy(model,predictor,alpha,temp,starts,max_steps,action_fraction,reward_mode,action_decay):
    env=CureDesignEnv(predictor,alpha,temp,max_steps,action_fraction,reward_mode,action_decay)
    finals=[]
    for x0 in starts:
        obs,info=env.reset(options={'x':x0})
        for _ in range(env.max_steps):
            act,_=model.predict(obs,deterministic=True)
            obs,_,_,_,info=env.step(act)
        finals.append({'score':float(info['score']),'feasible':bool(info['feasible']),**info['metrics']})
    scores=[m['score'] for m in finals]
    feasible=[m for m in finals if m['feasible']]
    return {'mean_score':float(np.mean(scores)),'feasible_rate':len(feasible)/len(finals),
            'best_feasible_time_min':min((m['hold_end_min'] for m in feasible),default=None),
            'mean_min_doc':float(np.mean([m['alpha_min'] for m in finals]))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--baseline',default='cure_cycle_1')
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',type=Path,default=Path('rl_results_v3'))
    p.add_argument('--alpha-req',type=float,default=0.82)
    p.add_argument('--t-limit',type=float,default=195.)
    p.add_argument('--reward-mode',choices=['v2','v3'],default='v3')
    p.add_argument('--max-steps',type=int,default=8)
    p.add_argument('--action-fraction',type=float,default=0.12)
    p.add_argument('--action-decay',type=float,default=0.5)
    p.add_argument('--eval-freq',type=int,default=250)
    p.add_argument('--eval-episodes',type=int,default=10)
    p.add_argument('--learning-starts',type=int,default=100)
    a=p.parse_args()
    if a.steps<1 or a.eval_freq<1 or a.eval_episodes<1:raise ValueError('steps/eval-freq/eval-episodes must be positive')
    a.output.mkdir(parents=True,exist_ok=True)
    baseline=get_baseline(a.cases,a.baseline)
    predictor=PersistentXPIDON(a.manifest,baseline)
    env=CureDesignEnv(predictor,a.alpha_req,a.t_limit,a.max_steps,a.action_fraction,a.reward_mode,a.action_decay)
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.env_checker import check_env
    check_env(env,warn=True)
    # Held-out, FIXED initial states, independent from training RNG.
    rng=np.random.default_rng(20261021)
    starts=rng.uniform(predictor.lo,predictor.hi,size=(a.eval_episodes,6))
    eval_path=a.output/'eval_history.csv'
    monitor=Monitor(env,filename=str(a.output/'train_monitor.csv'))
    model=SAC('MlpPolicy',monitor,verbose=1,seed=a.seed,tensorboard_log=str(a.output/'tensorboard'),
              learning_starts=min(a.learning_starts,max(1,a.steps//3)),batch_size=32)
    class DiagnosticCallback(BaseCallback):
        def __init__(self):super().__init__();self.last=0
        def _on_step(self):
            if self.num_timesteps-self.last>=a.eval_freq or self.num_timesteps==a.steps:
                result=evaluate_policy(self.model,predictor,a.alpha_req,a.t_limit,starts,a.max_steps,a.action_fraction,a.reward_mode,a.action_decay)
                row={'training_steps':self.num_timesteps,'unique_xpidon_evaluations':len(predictor.cache),**result}
                with eval_path.open('a',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(row))
                    if f.tell()==0:writer.writeheader()
                    writer.writerow(row)
                print('Evaluation:',row,flush=True)
                self.last=self.num_timesteps
            return True
    start=time.monotonic()
    model.learn(total_timesteps=a.steps,callback=DiagnosticCallback(),progress_bar=False)
    model.save(str(a.output/'sac_model'))
    result={'research_only':True,'baseline':a.baseline,'seed':a.seed,'steps':a.steps,
            'alpha_requirement':a.alpha_req,'temperature_limit_C':a.t_limit,
            'reward_mode':a.reward_mode,'max_steps_per_episode':a.max_steps,
            'action_fraction':a.action_fraction,'action_decay':a.action_decay,
            'eval_freq':a.eval_freq,'eval_episodes':a.eval_episodes,
            'unique_evaluations':len(predictor.cache),'elapsed_seconds':time.monotonic()-start,
            'notes':'Uses same frozen XPIDON checkpoints; FE validation pending. Optimization step is not physical process time.'}
    (a.output/'run_info.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Saved to',a.output,'unique predictions',len(predictor.cache))
if __name__=='__main__':main()
