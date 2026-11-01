from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from fe_reference.config import load_config
from fe_reference.verification import run_verification


def main():
    cfg=load_config();case=pd.read_csv(ROOT/'data'/'validation_cases.csv').iloc[0].to_dict()
    out=ROOT/'outputs'/'verification';out.mkdir(parents=True,exist_ok=True)
    mainres,mesh,time,energy,summary=run_verification(case,cfg)
    mesh.to_csv(out/'mesh_convergence.csv',index=False);time.to_csv(out/'time_convergence.csv',index=False)
    energy.to_csv(out/'energy_balance.csv',index=False);summary.to_csv(out/'verification_summary.csv',index=False)
    print('\nVerification summary:');print(summary.to_string(index=False))
    print('\nMesh convergence:');print(mesh.to_string(index=False))
    print('\nTime convergence:');print(time.to_string(index=False))
    if not bool(summary.all_checks_pass.iloc[0]):
        raise SystemExit('Verification criteria not all satisfied. Inspect outputs/verification.')

if __name__=='__main__':main()
