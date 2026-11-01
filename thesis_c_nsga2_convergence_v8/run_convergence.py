"""Run paired-budget NSGA-II convergence experiments (checkpoints within each 4000-eval run)."""
import argparse, csv, json, subprocess, sys, time
from pathlib import Path

BUDGETS=(1200,2000,4000)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    p.add_argument('--population',type=int,default=40)
    p.add_argument('--budget',type=int,default=4000)
    p.add_argument('--alpha-req',type=float,default=.82)
    p.add_argument('--manifest',default='../thesis_c_optimization/outputs/20261001T070827907494Z_preoptimization_xpidon/model_manifest.json')
    p.add_argument('--cases',default='../comparison/cases.csv')
    p.add_argument('--baseline',default='cure_cycle_1')
    p.add_argument('--output',type=Path,default=Path('./results/convergence'))
    p.add_argument('--skip-existing',action='store_true')
    args=p.parse_args()
    if args.population<4 or args.budget%args.population or args.budget<max(BUDGETS):p.error('Budget must be >=4000 and divisible by population')
    if any(b%args.population for b in BUDGETS):p.error('Checkpoint budgets must be divisible by population')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    (out/'convergence_config.json').write_text(json.dumps({'configuration':config,'checkpoints':BUDGETS,'method':'one continuous NSGA-II run per seed; paired checkpoints'},indent=2))
    script=Path(__file__).resolve().parent/'run_nsga2.py'
    for seed in args.seeds:
        run_dir=out/f'seed_{seed}'
        if run_dir.exists():
            if args.skip_existing and (run_dir/'summary.json').exists():
                print(f'Skipping completed {run_dir}',flush=True);continue
            raise FileExistsError(f'{run_dir} already exists. Use a new --output or --skip-existing for completed runs.')
        command=[sys.executable,str(script),'--budget',str(args.budget),'--population',str(args.population),'--seed',str(seed),'--alpha-req',str(args.alpha_req),'--manifest',str(Path(args.manifest).resolve()),'--cases',str(Path(args.cases).resolve()),'--baseline',args.baseline,'--report-every','5','--output',str(run_dir)]
        print('RUN:', ' '.join(command),flush=True)
        with (out/f'seed_{seed}.log').open('w') as log:
            proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in proc.stdout:
                print(f'[{seed}] {line}',end='',flush=True);log.write(line);log.flush()
            result=proc.wait()
        if result:raise RuntimeError(f'seed {seed} failed with code {result}; see {out}/seed_{seed}.log')
        # Update comparisons after each seed, including interrupted multi-seed campaigns.
        subprocess.run([sys.executable,str(Path(__file__).parent/'analyze_convergence.py'),'--root',str(out),'--checkpoints',*[str(x) for x in BUDGETS]],check=True)
    print('DONE:',out,flush=True)

if __name__=='__main__':main()
