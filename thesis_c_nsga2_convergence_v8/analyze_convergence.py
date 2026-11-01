"""Analyze real on-disk V7/V8 NSGA-II records at paired evaluation checkpoints."""
import argparse,csv,json,math,statistics
from pathlib import Path
import numpy as np
from optimization_problem import hypervolume_2d,pareto_front

def read_jsonl(path):
    with path.open() as f:return [json.loads(line) for line in f if line.strip()]

def checkpoint_metrics(rows,history,budget,seed):
    snapshot=rows[:budget]
    if len(snapshot)!=budget:raise ValueError(f'Seed {seed}: expected {budget} evaluations, found {len(snapshot)}')
    history_exact=next((h for h in history if h['evaluations']==budget),None)
    if history_exact is None:raise ValueError(f'Seed {seed}: no exact logged checkpoint at {budget}')
    feasible=[r for r in snapshot if r['feasible']]
    front=pareto_front(snapshot)
    out={'seed':seed,'budget':budget,'feasible_count':len(feasible),'feasible_rate':len(feasible)/budget,
         'pareto_count':len(front),'hypervolume':hypervolume_2d(snapshot),
         'best_feasible_time_min':min((float(r['hold_end_min']) for r in feasible),default=None),
         'best_feasible_doc':max((float(r['alpha_min']) for r in feasible),default=None),
         'elapsed_seconds':float(history_exact['elapsed_seconds']),
         'baseline_delta_time_percent':None}
    return out,front

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--checkpoints',type=int,nargs='+',default=[1200,2000,4000])
    args=p.parse_args();root=args.root.resolve()
    files=sorted(root.glob('seed_*/summary.json'))
    if not files:raise SystemExit(f'No completed runs under {root}')
    result_rows=[];fronts={};histories={}
    for summary_file in files:
        seed=int(summary_file.parent.name.removeprefix('seed_'))
        summary=json.loads(summary_file.read_text())
        records=read_jsonl(summary_file.parent/'all_evaluations.jsonl')
        history=read_jsonl(summary_file.parent/'convergence.jsonl')
        if len(records)!=summary['unique_evaluations']:raise ValueError(f'Seed {seed}: record count mismatch')
        if [r['evaluation'] for r in records]!=list(range(1,len(records)+1)):raise ValueError(f'Seed {seed}: evaluation sequence corrupted')
        histories[seed]=history
        for budget in args.checkpoints:
            if budget>len(records):continue
            result,front=checkpoint_metrics(records,history,budget,seed)
            base_time=float(summary['baseline']['hold_end_min'])
            if result['best_feasible_time_min'] is not None:
                result['baseline_delta_time_percent']=100*(base_time-result['best_feasible_time_min'])/base_time
            result_rows.append(result);fronts[(seed,budget)]=front
    if not result_rows:raise SystemExit('No checkpoints complete')
    keys=list(result_rows[0]);
    with (root/'checkpoint_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(result_rows)
    aggregate=[]
    metric_keys=['feasible_rate','hypervolume','best_feasible_time_min','best_feasible_doc','elapsed_seconds','baseline_delta_time_percent']
    for budget in sorted(set(r['budget'] for r in result_rows)):
        cohort=[r for r in result_rows if r['budget']==budget]
        item={'budget':budget,'n_seeds':len(cohort)}
        for k in metric_keys:
            values=[r[k] for r in cohort if r[k] is not None]
            item[k+'_mean']=statistics.mean(values) if values else None
            item[k+'_std']=statistics.stdev(values) if len(values)>1 else None
        aggregate.append(item)
    (root/'convergence_summary.json').write_text(json.dumps({'checkpoints':result_rows,'aggregated':aggregate,'notes':['All metrics derived from actual XPIDON prediction records.','Checkpoint budgets are nested within each independent 4000-evaluation seed run.','HV uses the fixed V7 reference (320 min, -DoC=0); use common reference and objective directions across all runs.','These are surrogate predictions; FEA validation remains pending.','Small standard deviations from few seeds are descriptive, not proof of statistical significance.']},indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,2,figsize=(12,8))
    colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i,(seed,history) in enumerate(histories.items()):
        c=colors[i%len(colors)]; xx=[h['evaluations'] for h in history]
        ax[0,0].plot(xx,[h['hypervolume'] for h in history],label=f'seed {seed}',color=c)
        ax[0,1].plot(xx,[h['best_feasible_time_min'] if h['best_feasible_time_min'] is not None else np.nan for h in history],label=f'seed {seed}',color=c)
        ax[1,0].plot(xx,[h['best_feasible_doc'] if h['best_feasible_doc'] is not None else np.nan for h in history],label=f'seed {seed}',color=c)
        ax[1,1].plot(xx,[h['elapsed_seconds'] for h in history],label=f'seed {seed}',color=c)
    for a,title,ylabel in zip(ax.flat,['Hypervolume','Best feasible time','Highest feasible minimum DoC','Cumulative runtime'],['Hypervolume (min × DoC)','Minutes','Degree of cure','Seconds']):
        for budget in args.checkpoints:a.axvline(budget,color='gray',ls=':',lw=.8)
        a.set(title=title,xlabel='Unique XPIDON evaluations',ylabel=ylabel);a.grid(alpha=.2)
    ax[0,0].legend()
    fig.suptitle('NSGA-II convergence across seeds — actual results');fig.tight_layout()
    fig.savefig(root/'convergence_comparison.png',dpi=190);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,6))
    last_budget=max(r['budget'] for r in result_rows)
    for i,seed in enumerate(histories):
        for b in args.checkpoints:
            front=fronts.get((seed,b),[])
            if front:
                ax.plot([x['hold_end_min'] for x in front],[x['alpha_min'] for x in front],'-o',markersize=2,alpha=.8,label=f'seed {seed} · {b}')
    ax.axhline(.82,color='gray',ls=':',label='Research constraint (0.82)')
    ax.set(xlabel='Heating-and-hold completion time (min)',ylabel='Minimum DoC at hold end',title='Feasible Pareto fronts at checkpoints')
    ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(root/'pareto_checkpoint_comparison.png',dpi=190);plt.close(fig)
    print(f'Analyzed {len(histories)} completed seed(s): {root}',flush=True)
    for a in aggregate:print('Checkpoint',a['budget'],'mean HV',a['hypervolume_mean'],'mean best time',a['best_feasible_time_min_mean'],flush=True)
if __name__=='__main__':main()
