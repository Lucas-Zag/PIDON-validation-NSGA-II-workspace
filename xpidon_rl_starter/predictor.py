"""Persistent frozen XPIDON inference; requires local checkpoints and repository."""
from pathlib import Path
import json, pickle
import numpy as np

PARAMS = ('ramp1_C_min','hold1_T_C','hold1_d_min','ramp2_C_min','hold2_T_C','hold2_d_min')
KEYS = (('r1_min','r1_max'),('ht1_min','ht1_max'),('hd1_min','hd1_max'),('r2_min','r2_max'),('ht2_min','ht2_max'),('hd2_min','hd2_max'))

class PersistentXPIDON:
    def __init__(self, manifest_path, baseline_case, time_step_s=10, spatial_points=81):
        import sys
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        self.workspace = Path(self.manifest['workspace']).resolve()
        self.root = self.workspace/'XPIDON-Composites-FV1'
        self.model_dir = Path(self.manifest['model_directory'])
        if not self.model_dir.is_absolute():
            self.model_dir = self.workspace/self.model_dir
        self.exp_dict = json.loads((self.root/'exp_params.json').read_text())
        sys.path.insert(0,str(self.root))
        from utils.phy_params import PhyParams
        from utils.exp_params import ExpParams
        from models.pidon import XPIDON
        self.exp = ExpParams(str(self.root/'exp_params.json'))
        phy = PhyParams(str(self.root/'phy_params.json'))
        self.baseline = dict(baseline_case)
        self.lo = np.array([self.exp_dict[a] for a,b in KEYS], dtype=np.float64)
        self.hi = np.array([self.exp_dict[b] for a,b in KEYS], dtype=np.float64)
        self.horizon_s = float(self.exp_dict['t_max'])
        self.positions = np.linspace(0,1,spatial_points)
        self.dt = float(time_step_s)
        self.tags = [str(t) for t in self.manifest['tags']]
        self.models = []
        prior = 0.
        for tag in self.tags:
            upper = int(tag)/10000.
            model = XPIDON(phy,self.exp,[9,50,50,50,100],[2,50,50,50,50,50,100],
                           [100,50,50,50,50,1],prior*self.horizon_s,upper*self.horizon_s)
            with (self.model_dir/f'xpidon_class_{tag}_T.pkl').open('rb') as f: wt=pickle.load(f)
            with (self.model_dir/f'xpidon_class_{tag}_a.pkl').open('rb') as f: wa=pickle.load(f)
            self.models.append((prior,upper,model,wt,wa))
            prior=upper
        if abs(prior-1.)>1e-7: raise ValueError('Last subdomain tag must end at 10000.')
        self.cache={}

    def _scale(self, x):
        e=self.exp_dict; r1,T1,h1,r2,T2,h2=map(float,x)
        b=self.baseline
        return np.asarray([(r1/60)*e['t_max']/e['T_max'],T1/e['T_max'],h1*60/e['t_max'],
            (r2/60)*e['t_max']/e['T_max'],T2/e['T_max'],h2*60/e['t_max'],
            b['hbot_W_m2K']/e['h_max'],b['htop_W_m2K']/e['h_max'],b['Lt_m']/e['l_max']],dtype=np.float32)

    def evaluate(self, x):
        import jax.numpy as jnp
        x=np.asarray(x,dtype=float)
        if x.shape!=(6,) or not np.all(np.isfinite(x)) or np.any(x<self.lo) or np.any(x>self.hi):
            raise ValueError('Candidate outside training bounds')
        key=tuple(x.tolist())
        if key in self.cache:return dict(self.cache[key])
        r1,T1,d1,r2,T2,d2=x
        T0=float(self.exp_dict['T_ini'])
        hold_end_s=60*((T1-T0)/r1+d1+(T2-T1)/r2+d2)
        if hold_end_s>=self.horizon_s:raise ValueError('No positive cooling time')
        times=np.unique(np.r_[np.arange(0,self.horizon_s,self.dt),self.horizon_s,hold_end_s,
                               [v*self.horizon_s for v,_,_,_,_ in self.models],
                               [v*self.horizon_s for _,v,_,_,_ in self.models]])
        vector=self._scale(x)
        ends=np.array([hi for _,hi,_,_,_ in self.models])
        indices=np.minimum(np.searchsorted(ends,times/self.horizon_s,side='right'),len(ends)-1)
        peak=-np.inf; alpha_hold=None
        for i,(lo,hi,model,wt,wa) in enumerate(self.models):
            sel=np.flatnonzero(indices==i)
            if not len(sel):continue
            q=np.column_stack([np.repeat(np.clip((times[sel]/self.horizon_s-lo)/(hi-lo),0,1),len(self.positions)),
                               np.tile(self.positions,len(sel))]).astype(np.float32)
            U=np.broadcast_to(vector,(len(q),9)).copy()
            temp=np.asarray(model.pred_T(wt,jnp.asarray(U),jnp.asarray(q))).reshape(len(sel),-1)/float(self.exp.T_scaler)
            peak=max(peak,float(np.max(temp)))
            if np.any(np.isclose(times[sel],hold_end_s,rtol=0,atol=1e-7)):
                j=np.flatnonzero(np.isclose(times[sel],hold_end_s,rtol=0,atol=1e-7))[0]
                qa=jnp.asarray(q[j*len(self.positions):(j+1)*len(self.positions)])
                ua=jnp.asarray(U[j*len(self.positions):(j+1)*len(self.positions)])
                alpha_hold=np.asarray(model.pred_a(wa,ua,qa)).ravel()
        if alpha_hold is None:raise RuntimeError('Hold-end alpha was not evaluated')
        if np.any(~np.isfinite(alpha_hold)) or not (np.all(alpha_hold>=0) and np.all(alpha_hold<=1)):
            raise ValueError('Unphysical alpha output')
        result={'hold_end_min':hold_end_s/60.,'alpha_min':float(np.min(alpha_hold)),
                'alpha_spread':float(np.ptp(alpha_hold)), 'peak_T_C':peak}
        self.cache[key]=result
        return dict(result)
