from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
FE_ROOT = WORKSPACE / "validation_fe_reference_v2"

if not FE_ROOT.exists():
    raise FileNotFoundError(f"FE repository not found: {FE_ROOT}")

sys.path.insert(0, str(FE_ROOT))

from fe_reference.config import load_config
from fe_reference.solver import solve_case
from fe_reference.diagnostics import key_curves, case_metrics


CASES_FILE = HERE / "outputs" / "cases_runtime.csv"
OUT_DIR = HERE / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if not CASES_FILE.exists():
        raise FileNotFoundError(
            f"{CASES_FILE} does not exist. Run prepare_cases.py first."
        )

    cfg = load_config()
    cases = pd.read_csv(CASES_FILE)

    all_curves = []
    all_metrics = []

    for case_number, (_, row) in enumerate(cases.iterrows(), start=1):
        case = row.to_dict()
        print(
            f"FE solving [{case_number}/{len(cases)}]: "
            f"{case['case_id']}"
        )
        result = solve_case(case, cfg)
        all_curves.append(key_curves(result))
        all_metrics.append(case_metrics(result))

    curves = pd.concat(all_curves, ignore_index=True)
    metrics = pd.DataFrame(all_metrics)

    curves_path = OUT_DIR / "fe_key_curves.csv"
    metrics_path = OUT_DIR / "fe_metrics.csv"

    curves.to_csv(curves_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    print(f"Saved: {curves_path}")
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
