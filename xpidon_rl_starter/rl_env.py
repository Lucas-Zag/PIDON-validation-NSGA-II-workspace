"""SAC design-refinement environment. Offline cure-cycle design, not autoclave control."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class CureDesignEnv(gym.Env):
    metadata = {'render_modes': []}
    def __init__(self, predictor, alpha_requirement=0.82, temperature_limit_C=195.,
                 max_steps=8, action_fraction=0.12, reward_mode='v3', action_decay=0.5):
        super().__init__()
        self.p=predictor
        self.alpha_req=float(alpha_requirement)
        self.temp_limit=float(temperature_limit_C)
        self.max_steps=int(max_steps)
        self.action_fraction=float(action_fraction)
        self.reward_mode=reward_mode
        self.action_decay=float(action_decay)
        if not 0<=self.alpha_req<=1 or self.max_steps<1 or self.temp_limit<=0:
            raise ValueError('Invalid constraints or maximum steps')
        if reward_mode not in ('v2','v3'): raise ValueError('Unknown reward mode')
        self.action_space=spaces.Box(-1.,1.,shape=(6,),dtype=np.float32)
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(10,),dtype=np.float32)
        self.x=None; self.metrics=None; self.steps=0
    def _obs(self):
        m=self.metrics
        norm=(self.x-self.p.lo)/(self.p.hi-self.p.lo)
        return np.asarray([*norm,m['hold_end_min']/313.34,m['alpha_min'],
                           m['peak_T_C']/250.,m['alpha_spread']],dtype=np.float32)
    def _feasible(self,m):
        return m['alpha_min']>=self.alpha_req and m['peak_T_C']<=self.temp_limit
    def _score(self,m):
        if self.reward_mode=='v2':
            return -m['hold_end_min']/250.-40*max(0.,self.alpha_req-m['alpha_min'])-0.2*max(0.,m['peak_T_C']-self.temp_limit)
        # Dimensionless constraint-aware score. Target values are research scenarios, not certified thresholds.
        doc_gap=max(0.,self.alpha_req-m['alpha_min']) / 0.02
        temp_gap=max(0.,m['peak_T_C']-self.temp_limit) / 5.
        time_cost=m['hold_end_min']/250.
        # Strong feasibility preference, but nonzero gradients toward feasible region.
        return float(-time_cost - 2.0*doc_gap - 2.0*temp_gap + (0.2 if self._feasible(m) else 0.))
    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        self.steps=0
        if options and 'x' in options:
            self.x=np.asarray(options['x'],dtype=float).copy()
            if self.x.shape!=(6,) or np.any(self.x<self.p.lo) or np.any(self.x>self.p.hi):
                raise ValueError('Invalid initial parameters')
        else:self.x=self.np_random.uniform(self.p.lo,self.p.hi)
        self.metrics=self.p.evaluate(self.x)
        return self._obs(),{'metrics':dict(self.metrics),'x':self.x.tolist(),'score':self._score(self.metrics),'feasible':self._feasible(self.metrics)}
    def step(self,action):
        a=np.asarray(action,dtype=float)
        if a.shape!=(6,) or not np.all(np.isfinite(a)):raise ValueError('Invalid action')
        old_score=self._score(self.metrics)
        step_scale=(1-self.action_decay*self.steps/max(1,self.max_steps-1))
        x_next=np.clip(self.x+np.clip(a,-1,1)*self.action_fraction*step_scale*(self.p.hi-self.p.lo),self.p.lo,self.p.hi)
        # Invalid geometry / cooling must not crash entire training.
        try:
            metrics=self.p.evaluate(x_next)
        except ValueError as e:
            self.steps+=1
            info={'metrics':dict(self.metrics),'x':self.x.tolist(),'score':old_score,
                  'feasible':self._feasible(self.metrics),'invalid_proposal':str(e)}
            return self._obs(),-2.,False,self.steps>=self.max_steps,info
        self.x=x_next;self.metrics=metrics;self.steps+=1
        score=self._score(metrics)
        return self._obs(),float(score-old_score),False,self.steps>=self.max_steps, {'metrics':dict(metrics),'x':self.x.tolist(),'score':score,'feasible':self._feasible(metrics)}
