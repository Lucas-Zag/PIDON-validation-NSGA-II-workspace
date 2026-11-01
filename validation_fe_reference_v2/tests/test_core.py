from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from fe_reference.config import load_config
from fe_reference.cure_cycle import CureCycle
from fe_reference.solver import solve_case
from fe_reference.diagnostics import monotonic_alpha_check


def case0():return pd.read_csv(ROOT/'data'/'validation_cases.csv').iloc[0].to_dict()

def test_complete_cycle():
    cfg=load_config();c=CureCycle.from_case(case0(),cfg)
    assert abs(c.temperature_C(0)-20)<1e-12
    assert abs(c.temperature_C(c.total_process_time_s)-20)<1e-12
    assert c.cooling_rate_C_min>0

def test_interface_and_alpha():
    cfg=load_config();r=solve_case(case0(),cfg,{'n_tool_elements':12,'n_part_elements':12,'output_dt_s':100,'max_step_s':50})
    # One shared FE node provides exact interface temperature continuity.
    assert r.mesh.interface_node in r.mesh.part_node_indices
    ok,_,amin,amax=monotonic_alpha_check(r)
    assert ok and amin>=0 and amax<=1
