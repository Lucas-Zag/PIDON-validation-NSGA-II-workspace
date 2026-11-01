#!/usr/bin/env python3
"""V9: post-process actual V8 NSGA-II run outputs (no model inference)."""
import argparse,csv,json,zipfile, tempfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PARAMS=['ramp1_C_min','hold1_T_C','hold1_d_min','ramp2_C_min','hold2_T_C','hold2_d_min']
COLORS={'Baseline':'#374151','Shortest feasible':'#0b6fa4','Highest DoC':'#d97706','Compromise':'#7c3aed'}

def read_json(p):return json.loads(p.read_text())
def read_jsonl(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def feasible(s,req):return bool(s.get('feasible',False)) and float(s['alpha_min'])>=req-1e-10

def nondominated(rows):
    # ascending time, monotonically increasing DoC needed to avoid domination by faster configurations
    ordered=sorted(rows,key=lambda s:(float(s['hold_end_min']),-float(s['alpha_min'])))
    out=[]; high=-float('inf')
    for s in ordered:
        d=float(s['alpha_min'])
        if d>high+1e-9:
            out.append(s);high=d
    return out

def choose(front):
    if not front:raise ValueError('No feasible solutions at requested alpha threshold')
    shortest=min(front,key=lambda s:s['hold_end_min']); highest=max(front,key=lambda s:s['alpha_min'])
    ts=np.array([s['hold_end_min'] for s in front]); ds=np.array([s['alpha_min'] for s in front]);
    dist=((ts-ts.min())/max(np.ptp(ts),1e-12))**2+((ds.max()-ds)/max(np.ptp(ds),1e-12))**2
    return {'Shortest feasible':shortest,'Highest DoC':highest,'Compromise':front[int(np.argmin(dist))]}

def style(ax,title,x,y):
    ax.set_title(title,fontsize=12,weight='bold');ax.set_xlabel(x);ax.set_ylabel(y);ax.grid(alpha=.25);ax.spines[['top','right']].set_visible(False)

def save(fig,path):fig.tight_layout();fig.savefig(path,dpi=300,bbox_inches='tight');plt.close(fig)

def profile(s,T0=20.):
    x=s['X'];r1,T1,d1,r2,T2,d2=map(float,x)
    if min(r1,r2)<=0 or T1<T0 or T2<T1:raise ValueError(f'Nonphysical heat profile: {x}')
    a=(T1-T0)/r1;b=a+d1;c=b+(T2-T1)/r2;d=c+d2
    return [0,a,b,c,d],[T0,T1,T1,T2,T2]

def mean_curve(ax, histories, field, label):
    max_gen=max(max(int(r['generation']) for r in h) for h in histories)
    mat=np.full((len(histories),max_gen),np.nan)
    for j,h in enumerate(histories):
        for r in h:
            v=r.get(field)
            if v is not None:mat[j,int(r['generation'])-1]=float(v)
    x=np.arange(1,max_gen+1)
    mean=np.nanmean(mat,axis=0);std=np.nanstd(mat,axis=0,ddof=1) if len(histories)>1 else np.zeros_like(mean)
    ax.plot(x,mean,label=label,lw=2);ax.fill_between(x,mean-std,mean+std,alpha=.16,label='±1 SD' if len(histories)>1 else None)
    return mean,std

def igd_plus(front,reference, t_range, doc_range):
    if not front or not reference:return None
    A=np.array([[x['hold_end_min']/t_range,-x['alpha_min']/doc_range] for x in front]);R=np.array([[x['hold_end_min']/t_range,-x['alpha_min']/doc_range] for x in reference]);
    # IGD+ uses max(a-r,0) for minimization
    return float(np.mean([np.min(np.linalg.norm(np.maximum(A-r,0),axis=1)) for r in R]))

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True,help='V8 convergence_3seeds directory or ZIP');p.add_argument('--output',default='./v9_results');p.add_argument('--alpha-req',type=float,default=0.82);args=p.parse_args()
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    temp=None; src=Path(args.input)
    if src.suffix.lower()=='.zip':
        temp=tempfile.TemporaryDirectory(); import os
        with zipfile.ZipFile(src) as z:
            for info in z.infolist():
                name=Path(info.filename)
                if name.is_absolute() or '..' in name.parts:raise ValueError('Unsafe ZIP path')
                if '__MACOSX' in name.parts or info.is_dir():continue
                target=Path(temp.name)/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(info))
        src=Path(temp.name)
    dirs=sorted([x.parent for x in src.rglob('all_evaluations.jsonl') if (x.parent/'convergence.jsonl').exists()])
    if not dirs:raise FileNotFoundError('Expected seed_*/all_evaluations.jsonl and convergence.jsonl')
    runs=[]; histories=[];all_feasible=[]
    for d in dirs:
        s=read_json(d/'summary.json'); h=read_jsonl(d/'convergence.jsonl');rows=read_jsonl(d/'all_evaluations.jsonl')
        if abs(float(s['alpha_requirement'])-args.alpha_req)>1e-9:raise ValueError('Run uses different DoC threshold: '+str(d))
        if s.get('objectives')!=['min heating_and_hold_time','max min_thickness_hold_end_DoC']:raise ValueError('Different objective definitions: '+str(d))
        runs.append((d,s,rows));histories.append(h);all_feasible += [dict(row,source_seed=s['seed']) for row in rows if feasible(row,args.alpha_req)]
    reference=nondominated(all_feasible); reps=choose(reference)
    baseline=runs[0][1]['baseline']; baseline=dict(baseline, X=[2.0,108,65,2.0,176,108]) # known Cycle 1 inputs; asserted below
    if abs(baseline['hold_end_min']-251.0)>1e-4:raise ValueError('Baseline mismatch; edit baseline inputs after review')
    fig,ax=plt.subplots(figsize=(8,5.4))
    for d,s,rows in runs:
        f=nondominated([r for r in rows if feasible(r,args.alpha_req)])
        ax.plot([r['hold_end_min'] for r in f],[r['alpha_min'] for r in f],lw=1,alpha=.37,label=f"Seed {s['seed']}")
    ax.plot([x['hold_end_min'] for x in reference],[x['alpha_min'] for x in reference],color='#0f766e',lw=2.5,label='Merged empirical reference')
    ax.scatter([baseline['hold_end_min']],[baseline['alpha_min']],marker='x',s=110,color=COLORS['Baseline'],label='Baseline Cycle 1',zorder=5)
    for name,sol in reps.items():ax.scatter([sol['hold_end_min']],[sol['alpha_min']],s=85,color=COLORS[name],edgecolors='white',linewidths=.9,label=name,zorder=6)
    ax.axhline(args.alpha_req,ls='--',color='gray',alpha=.7,label=f'DoC threshold {args.alpha_req}')
    style(ax,'Time–DoC Pareto front (XPIDON predictions)','Heating-and-hold time (min)','Minimum hold-end DoC');ax.legend(fontsize=8,ncol=2);save(fig,out/'pareto_time_doc.png')
    metrics=[('hypervolume','Hypervolume','hypervolume_convergence.png','Hypervolume (reported reference point)'),('best_feasible_time_min','Best feasible time','best_time_convergence.png','Best feasible time (min)'),('best_feasible_doc','Best feasible minimum DoC','best_doc_convergence.png','Minimum hold-end DoC')]
    for field,title,fn,ylabel in metrics:
        fig,ax=plt.subplots(figsize=(8,4.6));mean_curve(ax,histories,field,'Across-seed mean');style(ax,title+' vs generation','Generation',ylabel);ax.legend(loc='best');save(fig,out/fn)
    fig,ax=plt.subplots(figsize=(8,5))
    for name,s in [('Baseline',baseline),*reps.items()]:
        t,T=profile(s);ax.plot(t,T,label=f'{name} ({t[-1]:.1f} min)',lw=2,color=COLORS[name]);ax.plot(t[-1],T[-1],'.',color=COLORS[name])
    style(ax,'Autoclave air-temperature profiles (programmed)','Time (min)','Autoclave air temperature (°C)');ax.legend(fontsize=9);save(fig,out/'cure_profile_comparison.png')
    base_t=baseline['hold_end_min'];summary={'source_runs':[str(d) for d,_,_ in runs],'note':'Empirical merged nondominated front, not certified global optimum. All values are XPIDON predictions; FEA pending.','alpha_requirement':args.alpha_req,'reference_front_size':len(reference),'representative_solutions':{},'per_seed':[]}
    for name,s in reps.items():
        summary['representative_solutions'][name]={'source_seed':s.get('source_seed'), 'time_min':s['hold_end_min'],'minimum_doc':s['alpha_min'],'peak_temperature_C':s.get('peak_T_C'),'percent_time_reduction_vs_cycle1':100*(base_t-s['hold_end_min'])/base_t,'parameters':dict(zip(PARAMS,s['X']))}
    for d,s,rows in runs:
        own=nondominated([r for r in rows if feasible(r,args.alpha_req)])
        other=nondominated([r for r in all_feasible if r['source_seed']!=s['seed']]);
        summary['per_seed'].append({'seed':s['seed'],'evaluations':s['unique_evaluations'],'front_size':len(own),'IGD_plus_vs_leave_one_seed_out_reference':igd_plus(own,other,100,0.1),'note':'IGD+ uses fixed scales 100 min and 0.1 DoC; leave-one-seed-out reference is NOT an external ground truth'})
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    (out/'reference_pareto_front.json').write_text(json.dumps(reference,indent=2))
    (out/'representative_solutions.json').write_text(json.dumps(reps,indent=2))
    with (out/'representative_solutions.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['solution','seed','time_min','min_doc','peak_T_C','time_improvement_percent',*PARAMS]);
        for name,s in reps.items():writer.writerow([name,s.get('source_seed'),s['hold_end_min'],s['alpha_min'],s.get('peak_T_C'),100*(base_t-s['hold_end_min'])/base_t,*s['X']])
    print('Results:',out,'\nReference front:',len(reference));print(json.dumps(summary['representative_solutions'],indent=2))
    if temp:temp.cleanup()
if __name__=='__main__':main()
