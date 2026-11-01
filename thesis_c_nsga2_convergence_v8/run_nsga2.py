"""NSGA-II Time–DoC optimisation with reproducible on-disk live figures.
Plot files updated at report intervals. No real FE comparison is claimed.
"""
import argparse,csv,json,time,platform,sys
from pathlib import Path
import numpy as np
from optimization_problem import CureProblemDefinition,pareto_front,hypervolume_2d,representative_solutions,PARAMETERS
from predictor import PersistentXPIDON

DEFAULT_MANIFEST='../thesis_c_optimization/outputs/20261001T070827907494Z_preoptimization_xpidon/model_manifest.json'

def baseline_case(path,case_id):
    with open(path,newline='') as f:
        for row in csv.DictReader(f):
            if row.get('case_id')==case_id:
                return {k:(v if k in ('case_id','study','label','curve_label') else float(v)) for k,v in row.items() if v!=''}
    raise KeyError(f'Baseline {case_id!r} not found in {path}')

def plot_results(records,history,baseline,out):
    if not records:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    feasible=[r for r in records if r['feasible']]
    front=pareto_front(records)
    fig,axs=plt.subplots(2,3,figsize=(16,9))
    a=axs[0,0]
    a.scatter([r['hold_end_min'] for r in records],[r['alpha_min'] for r in records],s=8,alpha=.23,label='All')
    if feasible:a.scatter([r['hold_end_min'] for r in feasible],[r['alpha_min'] for r in feasible],s=9,alpha=.4,label='Feasible')
    if front:a.plot([r['hold_end_min'] for r in front],[r['alpha_min'] for r in front],'-o',markersize=2.5,label='Feasible Pareto')
    a.scatter([baseline['hold_end_min']],[baseline['alpha_min']],marker='x',s=70,label='Baseline')
    a.axhline(history[-1]['alpha_requirement'],ls='--',lw=1,label='Research threshold')
    a.set(xlabel='Heating + hold completion (min)',ylabel='Minimum DoC at hold end',title='Time–DoC Pareto front');a.legend(fontsize=7)
    x=[h['evaluations'] for h in history]
    axs[0,1].plot(x,[h['hypervolume'] for h in history]);axs[0,1].set(xlabel='Unique evaluations',ylabel='HV (min × DoC)',title='Feasible Hypervolume (reference: 320 min, 0 DoC)')
    axs[0,2].plot(x,[100*h['feasible_rate_cumulative'] for h in history],label='Cumulative')
    axs[0,2].plot(x,[100*h['feasible_rate_generation'] for h in history],label='Generation')
    axs[0,2].set(xlabel='Unique evaluations',ylabel='Feasible solutions (%)',ylim=(0,105),title='Feasible fraction');axs[0,2].legend(fontsize=8)
    axs[1,0].plot(x,[h['best_feasible_time_min'] for h in history],label='Min feasible time')
    axs[1,0].axhline(baseline['hold_end_min'],ls='--',label='Baseline time')
    axs[1,0].set(xlabel='Unique evaluations',ylabel='Time (min)',title='Best feasible time');axs[1,0].legend(fontsize=8)
    axs[1,1].plot(x,[h['best_feasible_doc'] for h in history],label='Max feasible DoC')
    axs[1,1].plot(x,[h['max_observed_doc'] for h in history],ls=':',label='Max observed DoC')
    axs[1,1].axhline(baseline['alpha_min'],ls='--',label='Baseline DoC')
    axs[1,1].set(xlabel='Unique evaluations',ylabel='Min thickness DoC',title='DoC progression');axs[1,1].legend(fontsize=8)
    axs[1,2].plot(x,[h['elapsed_seconds'] for h in history],label='Elapsed time')
    axs[1,2].set(xlabel='Unique evaluations',ylabel='Seconds',title='Execution time')
    for a in axs.flat:a.grid(alpha=.2)
    fig.suptitle('Thesis C | XPIDON + NSGA-II | Monitoring (research constraint)',fontweight='bold')
    fig.tight_layout();fig.savefig(out/'optimization_dashboard.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,6))
    ax.scatter([r['hold_end_min'] for r in records],[r['alpha_min'] for r in records],s=8,alpha=.22)
    if front:ax.plot([r['hold_end_min'] for r in front],[r['alpha_min'] for r in front],'-o',ms=3,label='Pareto')
    ax.scatter([baseline['hold_end_min']],[baseline['alpha_min']],marker='x',s=95,label='Baseline')
    ax.axhline(history[-1]['alpha_requirement'],ls='--',label='0.82 research threshold')
    ax.set(xlabel='Heating-and-hold completion time (min)',ylabel='Minimum thickness DoC at hold end',title='NSGA-II Time–DoC Trade-off');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'pareto_time_doc.png',dpi=180);plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default=DEFAULT_MANIFEST)
    p.add_argument('--cases',default='../comparison/cases.csv')
    p.add_argument('--baseline',default='cure_cycle_1')
    p.add_argument('--alpha-req',type=float,default=0.82)
    p.add_argument('--budget',type=int,default=1200)
    p.add_argument('--population',type=int,default=40)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--report-every',type=int,default=1)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.population<4 or args.budget<args.population or args.report_every<1:p.error('Require budget >= population >= 4 and report-every >= 1')
    if args.budget%args.population!=0:p.error('Use budget divisible by population for exact generation-wise budgets')
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.core.callback import Callback
    from pymoo.optimize import minimize
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    t0=time.perf_counter()
    base=baseline_case(args.cases,args.baseline)
    predictor=PersistentXPIDON(args.manifest,base)
    contract=CureProblemDefinition(args.alpha_req)
    base_vec=np.array([base[k] for k in PARAMETERS],dtype=float)
    base_metric=contract.evaluate(predictor,base_vec)
    predictor.cache.clear()
    rows=[];cache={};history=[];requested=0;generation_start=0
    with (out/'configuration.json').open('w') as f:json.dump({'args':{**vars(args),'output':str(args.output)},'objectives':['min hold_end_min','max min hold-end DoC'],'constraint':'min hold-end DoC >= alpha_req','peak_temperature':'reported only; not a hard constraint','hypervolume_reference_F':[320.,0.],'python':sys.version,'platform':platform.platform(),'baseline_case':args.baseline,'manifest':str(Path(args.manifest).resolve())},f,indent=2)
    def evaluate(x):
        nonlocal requested
        requested+=1
        key=tuple(np.asarray(x,dtype=float).tolist())
        if key in cache:return cache[key]
        if len(rows)>=args.budget:
            raise RuntimeError('Evaluation budget exceeded. No silent dummy penalty allowed.')
        tic=time.perf_counter()
        r=contract.evaluate(predictor,np.asarray(x))
        r={'X':list(key),**r,'evaluation':len(rows)+1,'evaluation_seconds':time.perf_counter()-tic}
        cache[key]=r;rows.append(r)
        with (out/'all_evaluations.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
        return r
    class Problem(ElementwiseProblem):
        def __init__(self):super().__init__(n_var=6,n_obj=2,n_ieq_constr=1,xl=predictor.lo,xu=predictor.hi)
        def _evaluate(self,x,out,*a,**k):
            r=evaluate(x);out['F']=r['F'];out['G']=r['G']
    class Monitor(Callback):
        def notify(self,alg):
            nonlocal generation_start
            feas=[r for r in rows if r['feasible']]
            gen_rows=rows[generation_start:];generation_start=len(rows)
            h={'generation':int(alg.n_gen),'evaluations':len(rows),'alpha_requirement':args.alpha_req,
               'feasible_count':len(feas),'feasible_rate_cumulative':len(feas)/max(1,len(rows)),
               'feasible_rate_generation':sum(r['feasible'] for r in gen_rows)/max(1,len(gen_rows)),
               'hypervolume':hypervolume_2d(rows),
               'best_feasible_time_min':min((r['hold_end_min'] for r in feas),default=None),
               'best_feasible_doc':max((r['alpha_min'] for r in feas),default=None),
               'max_observed_doc':max((r['alpha_min'] for r in rows),default=None),
               'elapsed_seconds':time.perf_counter()-t0,
               'avg_eval_seconds':float(np.mean([r['evaluation_seconds'] for r in rows]))}
            history.append(h)
            with (out/'convergence.jsonl').open('a') as f:f.write(json.dumps(h)+'\n')
            print(f"Gen {h['generation']:03} | {h['evaluations']}/{args.budget} evaluations | feasible {h['feasible_rate_cumulative']:.1%} | HV {h['hypervolume']:.3f} | best min-DoC {h['max_observed_doc']:.5f} | {h['elapsed_seconds']:.1f}s",flush=True)
            if h['generation']%args.report_every==0 or len(rows)>=args.budget:
                plot_results(rows,history,base_metric,out)
    ngen=args.budget//args.population
    minimize(Problem(),NSGA2(pop_size=args.population,sampling=FloatRandomSampling(),eliminate_duplicates=True),
             ('n_gen',ngen),seed=args.seed,callback=Monitor(),verbose=False)
    plot_results(rows,history,base_metric,out)
    front=pareto_front(rows);reps=representative_solutions(rows)
    (out/'pareto_front.json').write_text(json.dumps(front,indent=2))
    (out/'representative_solutions.json').write_text(json.dumps(reps,indent=2))
    base_t=base_metric['hold_end_min']
    summary={'method':'NSGA-II','seed':args.seed,'objectives':['min heating_and_hold_time','max min_thickness_hold_end_DoC'],
             'alpha_requirement':args.alpha_req,'temperature_constraint':None,
             'baseline':base_metric,'unique_evaluations':len(rows),'requested_evaluations':requested,
             'feasible_count':sum(r['feasible'] for r in rows),'feasible_fraction':np.mean([r['feasible'] for r in rows]),
             'hypervolume':hypervolume_2d(rows),'pareto_count':len(front),
             'representative_solutions':reps,'time_improvement_vs_baseline_percent':None if not reps else 100*(base_t-reps['shortest_time']['hold_end_min'])/base_t,
             'elapsed_seconds':time.perf_counter()-t0,'mean_evaluation_seconds':float(np.mean([r['evaluation_seconds'] for r in rows])) if rows else None,
             'validation_status':'XPIDON surrogate only; independent FEA pending',
             'caveat':'Peak temperature computed across whole model horizon; no material temperature cap applied. Hold-end time excludes cooling.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('\nFINAL:',json.dumps({k:v for k,v in summary.items() if k!='representative_solutions'},indent=2),flush=True)
if __name__=='__main__':main()
