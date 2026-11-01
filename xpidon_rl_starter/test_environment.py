"""Lightweight isolated reward/transition tests, no XPIDON weights required."""
import numpy as np
from rl_env import CureDesignEnv

class Dummy:
    lo=np.array([1.5,100,50,1.5,175,100],dtype=float)
    hi=np.array([3,120,70,3,185,110],dtype=float)
    def evaluate(self,x):
        x=np.asarray(x)
        return {'hold_end_min':220+(x[2]-55),'alpha_min':0.80+0.002*(x[4]-175),
                'peak_T_C':190+0.1*(x[4]-180),'alpha_spread':0.02}

p=Dummy();e=CureDesignEnv(p,max_steps=3)
a,meta=e.reset(seed=4)
assert a.shape==(10,)
for i in range(3):
    a,reward,term,trunc,info=e.step(np.zeros(6,dtype=np.float32))
    assert abs(reward)<1e-6 and not term and bool(trunc)==(i==2)
assert np.all(e.x>=p.lo) and np.all(e.x<=p.hi)
print('PASS: observation, transition, reward-delta, truncation')
