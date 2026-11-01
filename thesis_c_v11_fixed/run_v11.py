#!/usr/bin/env python3
"""V11 independent FE perturbation study; optional XPIDON, repeatable per-case timing.
Requires V10 integrated script and sibling repos, and prior V10 validation output.
"""
from __future__ import annotations
import argparse,csv,json,sys,time,hashlib,importlib.util
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=('ramp1_C_min','hold1_T_C','hold1_d_min','ramp2_C_min','hold2_T_C','hold2_d_min')
F=('hbot_W_m2K','htop_W_m2K','Lt_m','Lc_m','total_process_time_s')
DEFAULT_FRACS=[.01,.025]

def load(path):return json.loads(Path(path).read_text())
def save(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False))
def specs(src):
 with Path(src).open(newline='') as f: rows={r['case_id']:r for r in csv.DictReader(f)}
 for row in rows.values():
  for k in P+F:row[k]=float(row[k])
 return rows

def load_v10(path):
 spec=importlib.util.spec_from_file_location('v10_impl',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def perturbations(c,bounds,fracs):
 yield 'nominal',dict(c),'nominal',0,0
 for par in P:
  lo,hi=bounds[par];original=c[par]
  for frac in fracs:
   for sign in (-1,1):
    new=original+sign*frac*(hi-lo)
    if new<lo-1e-10 or new>hi+1e-10:continue # no clipped duplicates
    updated=dict(c);updated[par]=new
    yield f'{par}_{"plus" if sign>0 else "minus"}_{int(round(frac*1000))}permille',updated,par,frac,sign

def plot_fe(rows,out,alpha_req):
 fig,ax=plt.subplots(figsize=(9,5));order=['nominal']+[x for x in P]
 for i,k in enumerate(order):
  r=[x for x in rows if x['varied_parameter']==k]
  if not r:continue
  xs=np.full(len(r),i,dtype=float)+np.linspace(-.18,.18,len(r));ax.scatter(xs,[x['fe_min_doc'] for x in r],s=34,label=k if k!='nominal' else 'Nominal')
 ax.axhline(alpha_req,ls='--',color='gray',label=f'FEA DoC requirement {alpha_req}')
 ax.set_xticks(range(len(order)),['nominal','ramp 1','hold T1','hold d1','ramp 2','hold T2','hold d2'],rotation=20)
 ax.set_title('One-at-a-time FE perturbations — shortest feasible')
 ax.set_ylabel('Minimum DoC at hold-end');ax.grid(axis='y',alpha=.3);fig.tight_layout();fig.savefig(out/'FE_perturbation_doc.png',dpi=300);plt.close(fig)

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--workspace',type=Path,default=Path('..'))
 ap.add_argument('--v10-results',type=Path,required=True)
 ap.add_argument('--v10-script',type=Path,default=Path('../thesis_c_v10_integrated/run_v10.py'))
 ap.add_argument('--output',type=Path,required=True)
 ap.add_argument('--alpha-req',type=float,default=.82)
 ap.add_argument('--fraction',type=float,action='append',help='fraction of parameter width, e.g. 0.01; default 0.01 and 0.025')
 ap.add_argument('--all-representatives',action='store_true',help='also perturb Highest_DoC and Compromise')
 ap.add_argument('--with-xpidon',action='store_true',help='run XPIDON on same grid to compare perturbed FE cases (slower)')
 ap.add_argument('--benchmark-repeats',type=int,default=3,help='repeated paired FE and XPIDON calls on same grid')
 ap.add_argument('--time-step-s',type=float,default=20)
 ap.add_argument('--spatial-points',type=int,default=81)
 args=ap.parse_args();out=args.output.resolve();workspace=args.workspace.resolve();src=args.v10_results.resolve();v10=args.v10_script.resolve()
 if out.exists():raise FileExistsError(out)
 if args.benchmark_repeats<1:raise ValueError('benchmark-repeats must be positive')
 fracs=args.fraction or DEFAULT_FRACS
 if any(not 0<f<=.1 for f in fracs):raise ValueError('fraction must be 0 < x <= .1')
 v10data=load(src/'validation_summary.json');prev={x['case_id']:x for x in v10data}
 if 'Shortest_feasible' not in prev or 'fe_min_doc' not in prev['Shortest_feasible']:raise ValueError('Missing completed Shortest_feasible V10 result')
 cases=specs(src/'cases.csv');targets=['Shortest_feasible']+(['Highest_DoC','Compromise'] if args.all_representatives else [])
 for id in targets:
  if id not in cases:raise ValueError('Missing '+id)
 if not all((workspace/p).is_dir() for p in ['XPIDON-Composites-FV1','validation_fe_reference_v2','comparison']):raise FileNotFoundError('Expected 3 sibling repositories')
 v=load_v10(v10)
 fe_root=workspace/'validation_fe_reference_v2';cmp=workspace/'comparison';xp_root=workspace/'XPIDON-Composites-FV1'
 sys.path.insert(0,str(fe_root));sys.path.insert(0,str(cmp));sys.path.insert(0,str(xp_root))
 from fe_reference.config import load_config
 from fe_reference.solver import solve_case
 from utils.exp_params import ExpParams
 from utils.phy_params import PhyParams
 import run_xpidon_cases as cmp_xp
 cfg=load_config(fe_root/'config/reference_config.yaml');exp=ExpParams(str(xp_root/'exp_params.json'));phy=PhyParams(str(xp_root/'phy_params.json'))
 if not np.isclose(exp.t_max,cfg['cure_cycle']['total_process_time_s']):raise ValueError('Model/FE horizon mismatch')
 raw=load(xp_root/'exp_params.json')
 defs={'ramp1_C_min':('r1_min','r1_max'),'hold1_T_C':('ht1_min','ht1_max'),'hold1_d_min':('hd1_min','hd1_max'),'ramp2_C_min':('r2_min','r2_max'),'hold2_T_C':('ht2_min','ht2_max'),'hold2_d_min':('hd2_min','hd2_max')}
 bounds={k:(float(raw[a]),float(raw[b])) for k,(a,b) in defs.items()}
 if any(not np.isclose(c['Lc_m'],phy.part_len) for c in cases.values()):raise ValueError('Composite thickness mismatch')
 t0=float(cfg['initial_conditions']['temperature_C']);out.mkdir(parents=True);(out/'cases').mkdir();records=[]
 cmps=None;positions=np.linspace(0,1,args.spatial_points)
 if args.with_xpidon:
  cmp_xp.MODEL_DIR=xp_root/'outputs/models';cmps=cmp_xp.build_subdomains(phy,exp)
 for id in targets:
  ref=cases[id]
  for name,case,param,fraction,sign in perturbations(ref,bounds,fracs):
   cid=id+'__'+name
   end=v.hold_end(case,t0)
   if end>exp.t_max or end<0:raise ValueError('Hold-end out of range '+cid)
   start=time.perf_counter();fe=solve_case(case,cfg);fe_s=time.perf_counter()-start
   if fe.t_s[-1]<end-1e-7:raise ValueError('FE horizon too short')
   idx=fe.mesh.part_node_indices;z=(fe.mesh.z[idx]-fe.mesh.Lt_m)/fe.mesh.Lc_m
   # End alpha interpolation in time for each physical mesh node; then min over thickness
   fe_alpha_end=np.array([np.interp(end,fe.t_s,node) for node in fe.alpha]);doc=float(np.min(fe_alpha_end))
   # Original solver mesh directly for extrema, avoid interpolation optimism
   rec={'case_id':cid,'base_case':id,'varied_parameter':param,'fraction_of_training_range':fraction,'sign':sign,'hold_end_min':end/60,'fe_min_doc':doc,'fe_pass':doc>=args.alpha_req,'fe_solve_s':fe_s,'case_parameters':{k:case[k] for k in P}}
   if args.with_xpidon:
    times=np.unique(np.r_[np.arange(0,end,args.time_step_s),end]);Tx,ax,Ft,Fa,fsec,xsec=v.solve_fields(case,times,positions,workspace,cmp,cmps,phy,exp,cfg,None)
    rec.update(xp_min_doc=float(ax[-1].min()),xp_fe_doc_difference=float(ax[-1].min()-doc),temperature_RMSE_C=float(np.sqrt(np.mean((Tx-Ft)**2))),doc_RMSE=float(np.sqrt(np.mean((ax-Fa)**2))),xp_inference_s=xsec)
   records.append(rec);print(cid,'FE DoC',round(doc,7),'FE pass',doc>=args.alpha_req,flush=True)
   save(out/'cases'/f'{cid}.json',rec)
 save(out/'perturbation_results.json',records)
 with (out/'perturbation_results.csv').open('w',newline='') as f:
  keys=['case_id','base_case','varied_parameter','fraction_of_training_range','sign','hold_end_min','fe_min_doc','fe_pass','fe_solve_s','xp_min_doc','xp_fe_doc_difference','temperature_RMSE_C','doc_RMSE','xp_inference_s'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r.get(k) for k in keys} for r in records)
 plot_fe([r for r in records if r['base_case']=='Shortest_feasible'],out,args.alpha_req)
 summary={'status':'numerically_validated' if records else 'empty','alpha_threshold':args.alpha_req,'experiment':'One-at-a-time local parameter sensitivity (NOT joint uncertainty or probability of failure)','n_cases':len(records),'by_base':{}}
 for id in targets:
  r=[x for x in records if x['base_case']==id];summary['by_base'][id]={'n_tested':len(r),'n_failed':sum(not x['fe_pass'] for x in r),'min_fe_doc':min(x['fe_min_doc'] for x in r),'max_fe_doc':max(x['fe_min_doc'] for x in r),'all_tested_feasible':all(x['fe_pass'] for x in r)}
 # Benchmark full same-sized grid with FE and XPIDON, repeated without checkpoint reload
 cmp_xp.MODEL_DIR=xp_root/'outputs/models';subdomains=cmps if cmps is not None else cmp_xp.build_subdomains(phy,exp)
 ref=cases['Shortest_feasible'];end=v.hold_end(ref,t0);qt=np.unique(np.r_[np.arange(0,end,args.time_step_s),end]);bench=[]
 for rep in range(args.benchmark_repeats+1):
  Tx,ax,Ft,Fa,ts,ps=v.solve_fields(ref,qt,positions,workspace,cmp,subdomains,phy,exp,cfg,None)
  bench.append({'repeat':rep,'warmup':rep==0,'FE_s':ts,'XPIDON_s':ps,'FE_over_XPIDON_ratio':ts/ps,'grid_points':len(qt)*len(positions)})
 measured=bench[1:];summary['timing_benchmark']={'scope':'1D FE + XPIDON same full field grid; pretrained weights held in memory; each repeat includes FE solver call and XPIDON branch/query creation; first repeat treated as warmup','repeats':measured,'median_FE_s':float(np.median([x['FE_s'] for x in measured])),'median_XPIDON_s':float(np.median([x['XPIDON_s'] for x in measured])),'observed_FE_over_XPIDON_ratio':float(np.median([x['FE_s'] for x in measured])/np.median([x['XPIDON_s'] for x in measured])),'limitations':'single test case/1D solver, grid-size-dependent; not NSGA-II throughput; no loading/compilation startup in the reported medians'}
 save(out/'timing_repeats.json',bench);save(out/'v11_summary.json',summary)
 print(json.dumps(summary,indent=2)[:7500])
if __name__=='__main__':main()
