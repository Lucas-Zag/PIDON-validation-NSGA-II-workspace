"""Single objective/constraint contract for XPIDON + NSGA-II Thesis C.
No implied manufacturer certification. Uses minimum thickness DoC at hold end.
"""
from dataclasses import dataclass
import numpy as np

PARAMETERS=('ramp1_C_min','hold1_T_C','hold1_d_min','ramp2_C_min','hold2_T_C','hold2_d_min')

@dataclass(frozen=True)
class CureProblemDefinition:
    alpha_requirement:float=0.82
    def __post_init__(self):
        if not (0<self.alpha_requirement<1):raise ValueError('alpha_requirement must be between 0 and 1')
    def analyze(self,m):
        t=float(m['hold_end_min']); alpha=float(m['alpha_min']); temp=float(m['peak_T_C']); spread=float(m['alpha_spread'])
        if not np.isfinite([t,alpha,temp,spread]).all() or t<=0 or not 0<=alpha<=1 or spread<0:
            raise ValueError('Invalid XPIDON result')
        return {'F':[t,-alpha], 'G':[self.alpha_requirement-alpha],
                'feasible':bool(alpha>=self.alpha_requirement),
                'hold_end_min':t,'alpha_min':alpha,'peak_T_C':temp,'alpha_spread':spread}
    def evaluate(self,predictor,x):
        x=np.asarray(x,dtype=float)
        if x.shape!=(6,) or not np.isfinite(x).all() or np.any(x<predictor.lo) or np.any(x>predictor.hi):
            raise ValueError('Candidate outside XPIDON training range')
        return self.analyze(predictor.evaluate(x))

def pareto_front(records):
    feasible=sorted((r for r in records if r['feasible']),key=lambda r:(r['F'][0],r['F'][1]))
    front=[];best_second=float('inf')
    for r in feasible:
        if r['F'][1]<best_second-1.e-10:
            front.append(r);best_second=r['F'][1]
    return front

def hypervolume_2d(records,reference=(320.,0.)):
    """For minimization of (time,-DoC). Reference time=320min, -DoC=0.
    Fixed across experiments; points beyond reference give no area.
    """
    subset=[r for r in records if r['F'][0]<reference[0] and r['F'][1]<reference[1]]
    front=pareto_front(subset)
    area=0.;prior=reference[1]
    for r in front:
        x,y=r['F'];area+=(reference[0]-x)*max(0,prior-y);prior=min(prior,y)
    return float(area)

def representative_solutions(records):
    front=pareto_front(records)
    if not front:return {}
    fastest=min(front,key=lambda r:r['hold_end_min'])
    highest=max(front,key=lambda r:r['alpha_min'])
    times=np.asarray([r['hold_end_min'] for r in front]);alphas=np.asarray([r['alpha_min'] for r in front])
    tn=np.ptp(times);an=np.ptp(alphas)
    # normalized ideal-distance knee selection; no external or arbitrarily weighted objectives
    compromise=min(front,key=lambda r:((r['hold_end_min']-times.min())/(tn or 1))**2+((alphas.max()-r['alpha_min'])/(an or 1))**2)
    return {'shortest_time':fastest,'highest_doc':highest,'compromise':compromise}
