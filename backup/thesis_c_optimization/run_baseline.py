"""Launch isolated baseline inference; all outputs remain in this project."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace',type=Path,default=ROOT.parent)
    p.add_argument('--case-id',choices=['cure_cycle_1','cure_cycle_2'],default='cure_cycle_1')
    p.add_argument('--backend',choices=['xpidon','fe'],default='xpidon')
    p.add_argument('--python',default=sys.executable,help='Python executable in the existing model environment')
    p.add_argument('--tags',nargs='+',help='Explicit ordered paired checkpoint tags from one trained run')
    p.add_argument('--model-directory',type=Path)
    p.add_argument('--time-step-s',type=float,default=20)
    p.add_argument('--spatial-points',type=int,default=21)
    args = p.parse_args()
    if args.backend == 'xpidon' and not args.tags:
        p.error('XPIDON requires --tags to select one existing trained run')
    if not 0 < args.time_step_s < float('inf') or args.spatial_points < 3:
        p.error('Positive finite time step and at least 3 spatial points required')
    workspace = args.workspace.expanduser().resolve()
    model_dir = (args.model_directory.expanduser().resolve() if args.model_directory else
                 workspace/'XPIDON-Composites-FV1/outputs/models')
    cases = workspace/'comparison/cases.csv'
    if not cases.exists():
        p.error(f'Existing comparison cases not found: {cases}')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT/'outputs'/f'{stamp}_{args.backend}_{args.case_id}'
    output.mkdir(parents=True)
    request = {'workspace':str(workspace),'cases_file':str(cases),'case_id':args.case_id,
               'backend':args.backend,'tags':args.tags,'model_directory':str(model_dir),
               'time_step_s':args.time_step_s,'spatial_points':args.spatial_points,
               'output_directory':str(output)}
    request_path = output/'request.json'
    request_path.write_text(json.dumps(request,indent=2)+'\n')
    env = dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    with (output/'execution.log').open('w') as log:
        result = subprocess.run([args.python,'-B',str(ROOT/'model_worker.py'),
                                 '--request',str(request_path)],cwd=ROOT,env=env,
                                stdout=log,stderr=subprocess.STDOUT)
    print('Output:',output)
    if result.returncode:
        print('Baseline failed; see execution.log. No success summary was created.',file=sys.stderr)
        return result.returncode
    print((output/'summary.json').read_text())
    return 0


if __name__ == '__main__':
    sys.exit(main())
