#!/usr/bin/env python3
"""V10 reproducible validation: prepare four cases and compare independent FE vs XPIDON CSV fields.
No FE solver is bundled. NEVER treats XPIDON-only predictions as FE validation.
"""
import argparse, csv, json, hashlib, math, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PARAMS=['ramp1_C_min','hold1_T_C','hold1_d_min','ramp2_C_min','hold2_T_C','hold2_d_min']
LABELS=['Baseline','Shortest_feasible','Highest_DoC','Compromise']
BASE=[2.,108.,65.,2.,176.,108.]

def dump(p,o):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2,ensure_ascii=False))

def load(p):return json.loads(Path(p).read_text())

def case_time(x,t0=20):
    r1,T1,d1,r2,T2,d2=map(float,x)
    if not(r1>0 and r2>0 and T2>=T1>=t0 and d1>=0 and d2>=0): raise ValueError('Invalid heating profile')
    return (T1-t0)/r1+d1+(T2-T1)/r2+d2

def prepare(args):
    reps=load(args.representatives); dest=Path(args.output); dest.mkdir(parents=True,exist_ok=True)
    specs=[('Baseline',BASE,None)]+[(n,reps[n]['X'],reps[n]) for n in ['Shortest feasible','Highest DoC','Compromise']]
    index=[]
    for (label,x,source),safe in zip(specs,LABELS):
        x=list(map(float,x));end=case_time(x,args.initial_C)
        obj={'case_id':safe,'label':label,'parameters':dict(zip(PARAMS,x)), 'hold_end_min':end,
           'hbot_W_m2K':args.htc_bottom,'htop_W_m2K':args.htc_top,'tool_thickness_m':args.tool_thickness,
           'composite_thickness_m':args.composite_thickness,'initial_temperature_C':args.initial_C,
           'alpha_requirement':args.alpha_requirement,
           'protocol':'XPIDON vs independent FEA at matched time, position, material and boundary conditions',
           'model_prediction_unvalidated':None if source is None else {'minimum_hold_end_DoC':source['alpha_min'],'peak_part_temperature_C':source['peak_T_C']},
           'reference_prediction_horizon_s':18800}
        obj['case_sha256']=hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()
        dump(dest/'cases'/safe/'case.json',obj);index.append(obj)
    dump(dest/'cases_index.json',index)
    with (dest/'cases.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['case_id',*PARAMS,'hold_end_min','hbot_W_m2K','htop_W_m2K','tool_thickness_m','composite_thickness_m','initial_temperature_C'])
        for c in index:w.writerow([c['case_id'],*[c['parameters'][p] for p in PARAMS],c['hold_end_min'],c['hbot_W_m2K'],c['htop_W_m2K'],c['tool_thickness_m'],c['composite_thickness_m'],c['initial_temperature_C']])
    print('Prepared',len(index),'case specs at',dest)
    print('Export both model fields as cases/<case_id>/xpidon.csv and independent FEA as cases/<case_id>/fea.csv')

def read_field(path):
    if not path.exists():raise FileNotFoundError(path)
    with path.open(newline='') as f:
        rd=csv.DictReader(f); cols=set(rd.fieldnames or [])
        # Support existing XPIDON part_field convention; FE must use same normalized coordinate convention.
        if not {'t_s','x_norm','T_part_C','alpha'}<=cols:raise ValueError(f'{path}: expected t_s,x_norm,T_part_C,alpha; got {cols}')
        rows=[]
        for r in rd:
            rows.append([float(r[k]) for k in ('t_s','x_norm','T_part_C','alpha')])
    ar=np.array(rows,dtype=float)
    if ar.ndim!=2 or not len(ar) or not np.isfinite(ar).all():raise ValueError(f'{path}: empty/nonfinite field')
    if np.any(ar[:,1]<-1e-8) or np.any(ar[:,1]>1+1e-8) or np.any(ar[:,3]<-1e-8) or np.any(ar[:,3]>1+1e-8):raise ValueError(f'{path}: outside physical ranges')
    t=np.unique(ar[:,0]);x=np.unique(ar[:,1]); a=np.full((len(t),len(x),2),np.nan)
    ti=np.searchsorted(t,ar[:,0]);xi=np.searchsorted(x,ar[:,1]);
    if len(set(zip(ti,xi)))!=len(ar):raise ValueError(f'{path}: duplicate space-time point')
    a[ti,xi,:]=ar[:,2:4]
    if np.isnan(a).any():raise ValueError(f'{path}: incomplete rectangular time-space field')
    return t,x,a

def interpolate_field(t,x,a,qt,qx):
    if qt.min()<t.min()-1e-7 or qt.max()>t.max()+1e-7 or qx.min()<x.min()-1e-7 or qx.max()>x.max()+1e-7:
        raise ValueError('Time/space extrapolation forbidden: restrict requested comparison range')
    # first x, then t interpolation; each field dimension independently
    arr=np.empty((len(t),len(qx),2))
    for i in range(len(t)):
        for k in range(2):arr[i,:,k]=np.interp(qx,x,a[i,:,k])
    out=np.empty((len(qt),len(qx),2))
    for j in range(len(qx)):
        for k in range(2):out[:,j,k]=np.interp(qt,t,arr[:,j,k])
    return out

def metrics(a,b):
    delta=a-b
    return {'MAE':float(np.mean(np.abs(delta))),'RMSE':float(np.sqrt(np.mean(delta**2))),'bias_xpidon_minus_fea':float(np.mean(delta)),'max_absolute_error':float(np.max(np.abs(delta)))}

def plot_case(cid,qt,qx,xp,fe,end,out):
    # thick-axis min DoC and max temperature; defensible physical summaries
    mins=[xp[:,:,1].min(axis=1),fe[:,:,1].min(axis=1)]
    peaks=[xp[:,:,0].max(axis=1),fe[:,:,0].max(axis=1)]
    fig,ax=plt.subplots(2,1,figsize=(8,7),sharex=True)
    for v,lab in zip(peaks,['XPIDON','FEA']):ax[0].plot(qt/60,v,label=lab)
    for v,lab in zip(mins,['XPIDON','FEA']):ax[1].plot(qt/60,v,label=lab)
    ax[0].set_ylabel('Max part temperature (°C)');ax[1].set_ylabel('Minimum through-thickness DoC');ax[1].set_xlabel('Time (min)')
    ax[1].axhline(.82,color='gray',ls=':',label='Research threshold 0.82')
    for a in ax:a.axvline(end,ls='--',color='gray',alpha=.55);a.grid(alpha=.25);a.legend()
    fig.suptitle(f'{cid}: independent FEA validation');fig.tight_layout();fig.savefig(out/'temperature_doc_evolution.png',dpi=300);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,4));ax[0].plot(qt/60,np.mean(np.abs(xp[:,:,0]-fe[:,:,0]),axis=1));ax[1].plot(qt/60,np.mean(np.abs(xp[:,:,1]-fe[:,:,1]),axis=1));
    for a,title in zip(ax,['Temperature MAE over thickness','DoC MAE over thickness']):a.set_title(title);a.set_xlabel('Time (min)');a.grid(alpha=.25)
    ax[0].set_ylabel('°C');ax[1].set_ylabel('Absolute DoC');fig.tight_layout();fig.savefig(out/'error_evolution.png',dpi=300);plt.close(fig)

def verify(args):
    root=Path(args.input);specs=load(root/'cases_index.json');result=[]
    for c in specs:
        cid=c['case_id'];dire=root/'cases'/cid
        xp_path=dire/'xpidon.csv'; fe_path=dire/'fea.csv'
        if not fe_path.exists() or not xp_path.exists():
            result.append({'case_id':cid,'status':'PENDING','missing':[str(p) for p in [xp_path,fe_path] if not p.exists()]});continue
        try:
            xt,xx,xa=read_field(xp_path);ft,fx,fa=read_field(fe_path)
            end=c['hold_end_min']; end_s=end*60
            # evaluate shared physical heating interval [0, end] and include exact hold-end, no extrapolation
            upper=min(xt.max(),ft.max());lower=max(xt.min(),ft.min())
            if lower>1e-6 or upper<end_s-1e-6: raise ValueError('Both fields must cover [0, hold-end]')
            lo_x=max(xx.min(),fx.min());hi_x=min(xx.max(),fx.max())
            if lo_x>1e-7 or hi_x<1-1e-7:raise ValueError('Both fields must cover full normalized laminate thickness [0,1]')
            # Reference FEA nodes within shared interval: FE is ground truth, interpolation of XPIDON only.
            qt=np.unique(np.r_[ft[(ft>=0)&(ft<=end_s)],end_s]); qx=fx
            xp=interpolate_field(xt,xx,xa,qt,qx);fe=interpolate_field(ft,fx,fa,qt,qx)
            ma=metrics(xp[:,:,0],fe[:,:,0]);mb=metrics(xp[:,:,1],fe[:,:,1]);fe_end=float(fe[-1,:,1].min());xp_end=float(xp[-1,:,1].min())
            row={'case_id':cid,'status':'VALIDATED_NUMERICALLY','case_sha256':c['case_sha256'],
                 'samples':int(len(qt)*len(qx)), 'sampling_note':'FEA full-thickness spatial grid and native time samples; XPIDON bilinear interpolation',
                 'temperature_C':ma,'doc':mb,'minimum_doc_hold_end':{'xpidon':xp_end,'fea':fe_end,'difference_xpidon_minus_fea':xp_end-fe_end,'fea_meets_research_threshold':fe_end>=c['alpha_requirement']},
                 'peak_temperature_C_heating_hold':{'xpidon':float(xp[:,:,0].max()),'fea':float(fe[:,:,0].max()),'absolute_difference':float(abs(xp[:,:,0].max()-fe[:,:,0].max()))}}
            dest=Path(args.output)/cid;dest.mkdir(parents=True,exist_ok=True)
            plot_case(cid,qt,qx,xp,fe,end,dest);dump(dest/'metrics.json',row);result.append(row)
        except Exception as ex:
            result.append({'case_id':cid,'status':'ERROR','reason':str(ex)})
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);dump(out/'validation_summary.json',{'cases':result,'note':'Numerical comparison only. Case/material/FEA provenance must be reviewed before making physics-validation claims.'})
    with (out/'validation_summary.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['case_id','status','T_MAE_C','T_RMSE_C','alpha_MAE','alpha_RMSE','XPIDON_hold_end_minDoc','FEA_hold_end_minDoc','FEA_pass_0p82'])
        for r in result:
            if r['status']=='VALIDATED_NUMERICALLY':w.writerow([r['case_id'],r['status'],r['temperature_C']['MAE'],r['temperature_C']['RMSE'],r['doc']['MAE'],r['doc']['RMSE'],r['minimum_doc_hold_end']['xpidon'],r['minimum_doc_hold_end']['fea'],r['minimum_doc_hold_end']['fea_meets_research_threshold']])
            else:w.writerow([r['case_id'],r['status']])
    print(json.dumps([{'case_id':r['case_id'],'status':r['status']} for r in result],indent=2))

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--representatives',default='../thesis_c_nsga2_v9/example_results/representative_solutions.json');a.add_argument('--output',default='./results/v10_cases');a.add_argument('--initial-C',type=float,default=20);a.add_argument('--htc-bottom',type=float,default=70);a.add_argument('--htc-top',type=float,default=95);a.add_argument('--tool-thickness',type=float,default=.04);a.add_argument('--composite-thickness',type=float,default=.03);a.add_argument('--alpha-requirement',type=float,default=.82)
    b=sub.add_parser('verify');b.add_argument('--input',default='./results/v10_cases');b.add_argument('--output',default='./results/v10_validation')
    args=p.parse_args();prepare(args) if args.command=='prepare' else verify(args)
if __name__=='__main__':main()
