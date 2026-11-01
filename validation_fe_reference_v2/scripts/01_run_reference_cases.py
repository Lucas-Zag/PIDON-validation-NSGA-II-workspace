from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from fe_reference.config import load_config
from fe_reference.cure_cycle import CureCycle
from fe_reference.solver import solve_case
from fe_reference.diagnostics import key_curves,case_metrics,energy_balance


def main():
    cfg=load_config(); cases=pd.read_csv(ROOT/'data'/'validation_cases.csv')
    outi=ROOT/'outputs'/'inputs';outr=ROOT/'outputs'/'reference';outi.mkdir(parents=True,exist_ok=True);outr.mkdir(parents=True,exist_ok=True)
    sensors=[];fields=[];keys=[];metrics=[];energies=[]
    for _,r in cases.iterrows():
        case=r.to_dict()
        cycle=CureCycle.from_case(case,cfg);sensors.append(cycle.sensor_table(case['case_id'],int(cfg['cure_cycle']['sensor_count'])))
        print(f"Solving {case['case_id']} ...")
        sol=solve_case(case,cfg);fields.append(sol.full_field);keys.append(key_curves(sol));metrics.append(case_metrics(sol));energies.append(energy_balance(sol))
    pd.concat(sensors,ignore_index=True).to_csv(outi/'cure_cycle_sensors.csv',index=False)
    pd.concat(fields,ignore_index=True).to_csv(outr/'fe_full_field.csv',index=False)
    pd.concat(keys,ignore_index=True).to_csv(outr/'fe_key_curves.csv',index=False)
    pd.DataFrame(metrics).to_csv(outr/'fe_metrics.csv',index=False)
    pd.concat(energies,ignore_index=True).to_csv(outr/'energy_balance_all_cases.csv',index=False)
    print(pd.DataFrame(metrics).to_string(index=False))

if __name__=='__main__':main()
