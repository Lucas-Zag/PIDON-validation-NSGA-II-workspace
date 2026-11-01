#!/usr/bin/env python3
"""Run independent 1D FE and existing XPIDON for V9's four candidate cycles.
Requires sibling repos xpidoncompare, XPIDON-Composites-FV1, validation_fe_reference_v2.
Never retrains XPIDON; no FEA results invented.
"""
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np

PARAMS=('ramp1_C_min','hold1_T_C','hold1_d_min','ramp2_C_min','hold2_T_C','hold2_d_min')
FIXED=('hbot_W_m2K','htop_W_m2K','Lt_m','Lc_m','total_process_time_s')
NAMES={'Shortest feasible':'Shortest_feasible','Highest DoC':'Highest_DoC','Compromise':'Compromise'}

def load(p):return json.loads(Path(p).read_text())
def hold_end(c,t0):
    return 60*((c['hold1_T_C']-t0)/c['ramp1_C_min']+c['hold1_d_min']+(c['hold2_T_C']-c['hold1_T_C'])/c['ramp2_C_min']+c['hold2_d_min'])
def make_cases(reps,baseline):
    cases=[dict(baseline,case_id='Baseline',study='v10_validation',label='Baseline',curve_label='Baseline')]
    for name,cid in NAMES.items():
        r=reps[name]; case=dict(baseline,case_id=cid,study='v10_validation',label=name,curve_label=name)
        for key,val in zip(PARAMS,r['X']):case[key]=float(val)
        cases.append(case)
    return cases

def solve_fields(case,times,positions,workspace,comparison,subdomains,phy,exp,fe_cfg,fe_module):
    import jax.numpy as jnp
    from run_xpidon_cases import scale_case
    from fe_reference.solver import solve_case
    import pandas as pd
    fe_start=time.perf_counter();fe=solve_case(case,fe_cfg);fe_secs=time.perf_counter()-fe_start
    idx=fe.mesh.part_node_indices
    z=(fe.mesh.z[idx]-fe.mesh.Lt_m)/fe.mesh.Lc_m
    if not np.isclose(z[0],0) or not np.isclose(z[-1],1):raise ValueError('FE spatial nodes do not cover composite thickness')
    def interp_fe(field):
        temporal=np.stack([np.interp(times,fe.t_s,node) for node in field],axis=1)
        return np.stack([np.interp(positions,z,row) for row in temporal])
    if times[-1]>fe.t_s[-1]+1e-6:raise ValueError('FE simulation shorter than requested horizon')
    Tfe=interp_fe(fe.T_C[idx,:]); afe=interp_fe(fe.alpha)
    vector=scale_case(pd.Series(case),exp)
    Tx=np.empty_like(Tfe); ax=np.empty_like(afe)
    norm=times/float(exp.t_max)
    start=time.perf_counter()
    for i,sd in enumerate(subdomains):
        lo=float(sd['start']);hi=float(sd['end']);last=i==len(subdomains)-1
        mask=(norm>=lo-1e-12)&((norm<=hi+1e-12) if last else (norm<hi-1e-12))
        ids=np.flatnonzero(mask)
        if not len(ids):continue
        local=np.clip((norm[ids]-lo)/(hi-lo),0,1)
        query=np.column_stack([np.repeat(local,len(positions)),np.tile(positions,len(ids))])
        model=sd['model']
        temperature_flat=[];alpha_flat=[]
        for start_index in range(0,len(query),4096):
            q=jnp.asarray(query[start_index:start_index+4096]);branch=jnp.asarray(np.tile(vector,(len(q),1)))
            temperature_flat.extend(np.asarray(model.pred_T(sd['temperature_params'],branch,q)).reshape(-1).tolist())
            alpha_flat.extend(np.asarray(model.pred_a(sd['alpha_params'],branch,q)).reshape(-1).tolist())
        Tx[ids,:]=np.asarray(temperature_flat).reshape(len(ids),len(positions))/exp.T_scaler
        ax[ids,:]=np.asarray(alpha_flat).reshape(len(ids),len(positions))
    xp_secs=time.perf_counter()-start
    if not np.isfinite(Tx).all() or not np.isfinite(ax).all():raise ValueError('Nonfinite XPIDON fields')
    return Tx,ax,Tfe,afe,fe_secs,xp_secs

def summarize(cid,case,times,T,alpha,Ft,Fa,fe_secs,xp_secs,threshold):
    def errors(a,b):
        d=a-b;return {'MAE':float(np.mean(abs(d))),'RMSE':float(np.sqrt(np.mean(d*d))),'MaxAE':float(np.max(abs(d))),'bias':float(np.mean(d))}
    end_doc_x=float(alpha[-1,:].min());end_doc_f=float(Fa[-1,:].min())
    return {'case_id':cid,'hold_end_min':float(times[-1]/60),'xp_min_doc':end_doc_x,'fe_min_doc':end_doc_f,
            'xpidon_fe_doc_difference':end_doc_x-end_doc_f,'FEA_passes_0p82':bool(end_doc_f>=threshold),
            'xp_peak_part_T_C':float(T.max()),'fe_peak_part_T_C':float(Ft.max()),
            'T_error_C':errors(T,Ft),'DoC_error':errors(alpha,Fa),'FE_run_seconds':fe_secs,'XPIDON_inference_seconds':xp_secs,
            'comparison_interval':'heating-and-hold [0, hold-end] (not full cooling cycle)',
            'source':'Independent FE reference numerical solver vs frozen XPIDON checkpoints'}

def plot(cid,times,T,alpha,Ft,Fa,end,out,threshold):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,1,figsize=(8.5,7),sharex=True)
    axs[0].plot(times/60,T.max(axis=1),label='XPIDON',lw=2)
    axs[0].plot(times/60,Ft.max(axis=1),label='Independent FE',lw=2,ls='--')
    axs[0].set_ylabel('Maximum composite temperature (°C)')
    axs[1].plot(times/60,alpha.min(axis=1),label='XPIDON',lw=2)
    axs[1].plot(times/60,Fa.min(axis=1),label='Independent FE',lw=2,ls='--')
    axs[1].axhline(threshold,color='grey',ls=':',label=f'DoC ≥ {threshold}')
    axs[1].set_ylabel('Minimum thickness DoC');axs[1].set_xlabel('Time (min)')
    for a in axs:a.grid(alpha=.2);a.legend(fontsize=9)
    fig.suptitle(f'{cid} — Hold-end validation');fig.tight_layout();fig.savefig(out/'temperature_doc_comparison.png',dpi=300);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    axs[0].plot(times/60,np.mean(abs(T-Ft),axis=1));axs[0].set(ylabel='Temperature MAE across thickness (°C)',xlabel='Time (min)')
    axs[1].plot(times/60,np.mean(abs(alpha-Fa),axis=1));axs[1].set(ylabel='DoC MAE across thickness',xlabel='Time (min)')
    for a in axs:a.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'error_evolution.png',dpi=300);plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parent.parent)
    p.add_argument('--representatives',type=Path,default=None)
    p.add_argument('--output',type=Path,default=Path('./results/v10_integrated'))
    p.add_argument('--alpha-req',type=float,default=.82)
    p.add_argument('--time-step-s',type=float,default=10)
    p.add_argument('--spatial-points',type=int,default=81)
    args=p.parse_args();workspace=args.workspace.resolve();out=args.output.resolve()
    if out.exists():raise FileExistsError(f'Output exists: {out}; choose a new directory to protect existing experiments')
    cmp=workspace/'comparison'; fe_root=workspace/'validation_fe_reference_v2'; xp_root=workspace/'XPIDON-Composites-FV1'
    for path in (cmp,fe_root,xp_root):
        if not path.is_dir():raise FileNotFoundError(f'Missing sibling directory: {path}')
    repfile=args.representatives or (workspace/'thesis_c_nsga2_v9/example_results/representative_solutions.json')
    reps=load(repfile)
    with (cmp/'cases.csv').open(newline='') as f:
        baseline=next((row for row in csv.DictReader(f) if row['case_id']=='cure_cycle_1'),None)
    if baseline is None:raise ValueError('comparison/cases.csv has no cure_cycle_1')
    for k in PARAMS+FIXED:baseline[k]=float(baseline[k])
    cases=make_cases(reps,baseline)
    sys.path.insert(0,str(fe_root));sys.path.insert(0,str(cmp))
    from fe_reference.config import load_config
    import run_xpidon_cases as compare_xp
    from utils.phy_params import PhyParams
    from utils.exp_params import ExpParams
    phy=PhyParams(str(xp_root/'phy_params.json'));exp=ExpParams(str(xp_root/'exp_params.json'))
    cfg=load_config(fe_root/'config/reference_config.yaml')
    if not np.isclose(baseline['Lc_m'],phy.part_len):raise ValueError('Composite thickness mismatches trained XPIDON')
    if not np.isclose(baseline['total_process_time_s'],exp.t_max):raise ValueError('Model horizon mismatch')
    if not np.isclose(float(cfg['cure_cycle']['total_process_time_s']),exp.t_max):raise ValueError('FE time horizon mismatch')
    compare_xp.MODEL_DIR=xp_root/'outputs/models'
    subdomains=compare_xp.build_subdomains(phy,exp)
    out.mkdir(parents=True);(out/'cases').mkdir()
    with (out/'cases.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['study','case_id',*PARAMS,*FIXED,'label','curve_label']);writer.writeheader();writer.writerows(cases)
    (out/'provenance.json').write_text(json.dumps({'representatives':str(repfile.resolve()),'workspace':str(workspace),'FE_config':str((fe_root/'config/reference_config.yaml').resolve()),'time_step_s':args.time_step_s,'spatial_points':args.spatial_points,'research_Doc_requirement':args.alpha_req},indent=2))
    summary=[];pos=np.linspace(0,1,args.spatial_points)
    for case in cases:
        cid=case['case_id'];end=hold_end(case,float(cfg['initial_conditions']['temperature_C']))
        if end>exp.t_max or end<=0:raise ValueError(f'{cid}: hold_end outside model horizon')
        times=np.unique(np.r_[np.arange(0,end,args.time_step_s),end]);path=out/'cases'/cid;path.mkdir()
        print(f'Running {cid}: {len(times)} times × {len(pos)} positions',flush=True)
        Tx,ax,Tfe,afe,fe_secs,xp_secs=solve_fields(case,times,pos,workspace,cmp,subdomains,phy,exp,cfg,None)
        m=summarize(cid,case,times,Tx,ax,Tfe,afe,fe_secs,xp_secs,args.alpha_req)
        (path/'metrics.json').write_text(json.dumps(m,indent=2));summary.append(m)
        plot(cid,times,Tx,ax,Tfe,afe,end,path,args.alpha_req)
        with (path/'part_field.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['t_s','x_norm','T_XPIDON_C','alpha_XPIDON','T_FE_C','alpha_FE'])
            for i,t in enumerate(times):
                for j,x in enumerate(pos):w.writerow([t,x,Tx[i,j],ax[i,j],Tfe[i,j],afe[i,j]])
    (out/'validation_summary.json').write_text(json.dumps(summary,indent=2))
    with (out/'validation_summary.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['case_id','hold_end_min','xp_min_doc','fe_min_doc','FEA_passes_0p82','T_MAE_C','T_RMSE_C','alpha_MAE','alpha_RMSE','FE_run_seconds','XPIDON_inference_seconds'])
        for r in summary:w.writerow([r['case_id'],r['hold_end_min'],r['xp_min_doc'],r['fe_min_doc'],r['FEA_passes_0p82'],r['T_error_C']['MAE'],r['T_error_C']['RMSE'],r['DoC_error']['MAE'],r['DoC_error']['RMSE'],r['FE_run_seconds'],r['XPIDON_inference_seconds']])
    print('Completed:',out)
if __name__=='__main__':main()
