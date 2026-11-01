"""Thickness-field diagnostic for XPIDON-Composites-FV1; does not optimize or train."""
import argparse
import csv
import json
import pickle
import re
import sys
from pathlib import Path
import numpy as np


def query_grid(times, positions):
    tt, xx = np.meshgrid(times, positions, indexing='ij')
    return np.column_stack((tt.ravel(), xx.ravel()))


def metrics(T, alpha):
    T, alpha = np.asarray(T), np.asarray(alpha)
    if T.ndim != 2 or T.shape != alpha.shape or T.shape[1] < 2:
        raise ValueError('Expected matching [time, position] matrices with >=2 positions')
    if not np.isfinite(T).all() or not np.isfinite(alpha).all():
        raise ValueError('Nonfinite model predictions')
    return dict(delta_T=T.max(axis=1)-T.min(axis=1),
                delta_alpha=alpha.max(axis=1)-alpha.min(axis=1),
                alpha_min=alpha.min(axis=1), alpha_max=alpha.max(axis=1))


def write_csv(path, header, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def run(args):
    root = Path(args.repo).resolve()
    case = json.loads(Path(args.case).read_text())
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory is not empty; choose a new --out')
    if args.nx < 3 or args.nx % 2 != 1 or args.dt <= 0:
        raise ValueError('Use an odd --nx >=3 and positive --dt')
    if args.demo:
        times = np.arange(0., 601., args.dt)
        x = np.linspace(0., 1., args.nx)
        q = query_grid(times, x)
        T = (20+q[:,0]*0.1+10*q[:,1]).reshape(len(times),len(x))
        a = (0.05+q[:,0]*0.001+0.02*q[:,1]).reshape(T.shape)
        thickness = 0.03
        checks = []; jumps = []; backend = 'synthetic, not XPIDON'
        hold_end = 300.
    else:
        sys.path.insert(0, str(root))
        import jax
        import jax.numpy as jnp
        from models.pidon import XPIDON
        from utils.phy_params import PhyParams
        from utils.exp_params import ExpParams
        phy = PhyParams(str(root/'phy_params.json'))
        exp = ExpParams(str(root/'exp_params.json'))
        p = np.array([case[k] for k in ['ramp1_C_min','hold1_T_C','hold1_min','ramp2_C_min','hold2_T_C','hold2_min','htc_bottom','htc_top','tool_thickness_m']], dtype=float)
        u = p*np.array([exp.T_scaler*exp.t_max/60, exp.T_scaler,60/exp.t_max,exp.T_scaler*exp.t_max/60,exp.T_scaler,60/exp.t_max,exp.h_scaler,exp.h_scaler,exp.len_scaler])
        if np.any(u<np.asarray(exp.minvals)-1e-7) or np.any(u>np.asarray(exp.maxvals)+1e-7):
            raise ValueError('Case outside configured training parameter ranges')
        hold_end = ((p[1]-exp.T_ini/exp.T_scaler)/p[0]+p[2]+(p[4]-p[1])/p[3]+p[5])*60
        if hold_end > exp.t_max:
            raise ValueError('Hold end exceeds trained time horizon')
        model_dir = Path(args.models).resolve() if args.models else root/'outputs/models'
        ts = {re.fullmatch(r'xpidon_class_(\d+)_T.pkl',f.name).group(1):f for f in model_dir.glob('xpidon_class_*_T.pkl') if re.fullmatch(r'xpidon_class_(\d+)_T.pkl',f.name)}
        aa = {re.fullmatch(r'xpidon_class_(\d+)_a.pkl',f.name).group(1):f for f in model_dir.glob('xpidon_class_*_a.pkl') if re.fullmatch(r'xpidon_class_(\d+)_a.pkl',f.name)}
        if args.tags:
            tags=args.tags
        else:
            if len(ts)!=1 or set(ts)!=set(aa):
                raise ValueError('Supply explicit --tags in chronological order for one trained run; do not mix historical models')
            tags=list(ts)
        ends=np.array([int(t)/10000 for t in tags])
        if not len(ends) or np.any(np.diff(ends)<=0) or ends[-1]!=1.0 or ends[0]<=0:
            raise ValueError('Tags must increase and end at 10000')
        if any(t not in ts or t not in aa for t in tags):
            raise ValueError('Missing matching T/alpha model files')
        models=[]; prev=0.
        for tag,end in zip(tags,ends):
            md=XPIDON(phy,exp,[9,50,50,50,100],[2,50,50,50,50,50,100],[100,50,50,50,50,1],prev*exp.t_max,end*exp.t_max)
            # Only load your own trusted locally trained pickle weights.
            with open(ts[tag],'rb') as f: tp=pickle.load(f)
            with open(aa[tag],'rb') as f: ap=pickle.load(f)
            models.append((prev*exp.t_max,end*exp.t_max,md,tp,ap)); prev=end
        x=np.linspace(0.,1.,args.nx); thickness=float(phy.part_len)
        times=np.unique(np.r_[np.arange(0.,exp.t_max,args.dt),exp.t_max,hold_end,ends*exp.t_max])
        T=np.empty((len(times),len(x))); a=np.empty_like(T); checks=[]; jumps=[]
        def predict(item, physical_times, positions):
            lo,hi,md,tp,ap=item
            q=query_grid((np.asarray(physical_times)-lo)/(hi-lo),positions)
            U=jnp.asarray(np.tile(u,(len(q),1)))
            Y=jnp.asarray(q)
            return (np.asarray(md.pred_T(tp,U,Y))/exp.T_scaler).reshape(len(physical_times),len(positions)), np.asarray(md.pred_a(ap,U,Y)).reshape(len(physical_times),len(positions))
        assignment=np.searchsorted(ends*exp.t_max,times,side='left')
        for i,item in enumerate(models):
            ids=np.flatnonzero(assignment==i)
            T[ids],a[ids]=predict(item,times[ids],x)
            k=ids[len(ids)//2]
            # Independent singleton calls verify flattened-grid ordering and same-time rows.
            for j in [0,len(x)//2,len(x)-1]:
                st,sa=predict(item,[times[k]],[x[j]])
                np.testing.assert_allclose(T[k,j],st[0,0],rtol=1e-5,atol=1e-3)
                np.testing.assert_allclose(a[k,j],sa[0,0],rtol=1e-5,atol=1e-5)
                checks.append(dict(time_seconds=float(times[k]),x_norm=float(x[j]),temperature_C=float(st[0,0]),doc=float(sa[0,0])))
        for left,right in zip(models[:-1],models[1:]):
            lT,la=predict(left,[left[1]],x); rT,ra=predict(right,[right[0]],x)
            jumps.append(dict(time_seconds=left[1],max_temperature_jump_C=float(abs(lT-rT).max()),max_doc_jump=float(abs(la-ra).max())))
        backend=str(jax.default_backend())+' '+str(jax.devices())
    m=metrics(T,a); out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'field.npz',time_seconds=times,x_norm=x,z_m=x*thickness,temperature_C=T,doc=a)
    write_csv(out/'field.csv',['time_seconds','time_minutes','x_norm','z_mm','temperature_C','doc'],((t,t/60,xx,xx*thickness*1000,T[i,j],a[i,j]) for i,t in enumerate(times) for j,xx in enumerate(x)))
    write_csv(out/'uniformity.csv',['time_seconds','delta_T_C','delta_doc','doc_min','doc_max'],zip(times,m['delta_T'],m['delta_alpha'],m['alpha_min'],m['alpha_max']))
    selected=sorted(set([int(np.argmax(m['delta_T'])),int(np.argmax(m['delta_alpha'])),int(np.argmin(abs(times-hold_end))),len(times)-1]))
    write_csv(out/'spatial_profiles.csv',['time_seconds','x_norm','z_mm','temperature_C','doc'],((times[i],xx,xx*thickness*1000,T[i,j],a[i,j]) for i in selected for j,xx in enumerate(x)))
    endidx=int(np.argmin(abs(times-hold_end)))
    summary=dict(mode='synthetic' if args.demo else 'XPIDON',backend=backend,case=case,shape=list(T.shape),axes=['time','thickness_position'],hold_end_seconds=hold_end,maximum_spatial_temperature_difference_C=float(m['delta_T'].max()),temperature_difference_peak_time_min=float(times[np.argmax(m['delta_T'])]/60),maximum_spatial_doc_difference=float(m['delta_alpha'].max()),hold_end_min_doc=float(m['alpha_min'][endidx]),hold_end_doc_difference=float(m['delta_alpha'][endidx]),full_horizon_final_min_doc=float(m['alpha_min'][-1]),full_horizon_final_doc_difference=float(m['delta_alpha'][-1]),singleton_checks=checks,subdomain_boundary_jumps=jumps,warning='Predicted thickness uniformity only; no defect or quality threshold is inferred.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for j in [0,len(x)//2,len(x)-1]:
        axs[0,0].plot(times/60,T[:,j],label=f'x={x[j]:g}')
        axs[0,1].plot(times/60,a[:,j],label=f'x={x[j]:g}')
    axs[0,0].set_ylabel('Temperature (C)');axs[0,1].set_ylabel('Degree of cure')
    axs[1,0].plot(times/60,m['delta_T']);axs[1,0].set_ylabel('Spatial temperature range (C)')
    axs[1,1].plot(times/60,m['delta_alpha']);axs[1,1].set_ylabel('Spatial DoC range');axs[1,1].set_ylim(0,max(1e-6,float(m['delta_alpha'].max())*1.1))
    for ax in axs.flat: ax.set_xlabel('Time (min)');ax.grid();ax.axvline(hold_end/60,color='gray',linestyle='--')
    axs[0,0].legend();axs[0,1].legend();fig.suptitle(summary['mode']+' thickness diagnostic');fig.tight_layout();fig.savefig(out/'uniformity.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,field,label in zip(axs,[T,a],['Temperature (C)','Degree of cure']):
        im=ax.pcolormesh(times/60,x*thickness*1000,field.T,shading='auto');fig.colorbar(im,ax=ax,label=label);ax.set_xlabel('Time (min)');ax.set_ylabel('Thickness position (mm)')
    fig.tight_layout();fig.savefig(out/'fields.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['singleton_checks','case']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',default='.');p.add_argument('--case',default=str(Path(__file__).with_name('case_example.json')))
    p.add_argument('--models');p.add_argument('--tags',nargs='+');p.add_argument('--nx',type=int,default=81);p.add_argument('--dt',type=float,default=10)
    p.add_argument('--out',required=True);p.add_argument('--demo',action='store_true')
    run(p.parse_args())
