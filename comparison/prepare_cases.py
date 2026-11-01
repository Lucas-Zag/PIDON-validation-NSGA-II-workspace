from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
XPIDON_ROOT = WORKSPACE / "XPIDON-Composites-FV1"

MANUAL_CASES_FILE = HERE / "cases.csv"
CONFIG_FILE = HERE / "comparison_config.json"
OUT_DIR = HERE / "outputs"
RUNTIME_CASES_FILE = OUT_DIR / "cases_runtime.csv"
RANGE_REPORT_FILE = OUT_DIR / "case_range_report.csv"

REQUIRED_COLUMNS = [
    "study",
    "case_id",
    "ramp1_C_min",
    "hold1_T_C",
    "hold1_d_min",
    "ramp2_C_min",
    "hold2_T_C",
    "hold2_d_min",
    "hbot_W_m2K",
    "htop_W_m2K",
    "Lt_m",
    "Lc_m",
    "total_process_time_s",
    "label",
    "curve_label",
]

RANGE_MAP = {
    "ramp1_C_min": ("r1_min", "r1_max"),
    "hold1_T_C": ("ht1_min", "ht1_max"),
    "hold1_d_min": ("hd1_min", "hd1_max"),
    "ramp2_C_min": ("r2_min", "r2_max"),
    "hold2_T_C": ("ht2_min", "ht2_max"),
    "hold2_d_min": ("hd2_min", "hd2_max"),
    "hbot_W_m2K": ("hcb_min", "hcb_max"),
    "htop_W_m2K": ("hct_min", "hct_max"),
    "Lt_m": ("lt_min", "lt_max"),
}


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_columns(cases: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(cases.columns)
    if missing:
        raise ValueError(
            "cases.csv is missing columns: "
            + ", ".join(sorted(missing))
        )

    duplicated = cases["case_id"].duplicated(keep=False)
    if duplicated.any():
        values = cases.loc[duplicated, "case_id"].tolist()
        raise ValueError(f"Duplicate case_id values: {values}")


def validate_study_rules(cases: pd.DataFrame) -> None:
    cycle = cases[cases["study"] == "cure_cycle"]
    if not cycle.empty:
        if len(cycle) != 2:
            raise ValueError(
                "The cure_cycle study must contain exactly two cases."
            )
        fixed = [
            "hbot_W_m2K",
            "htop_W_m2K",
            "Lt_m",
            "Lc_m",
            "total_process_time_s",
        ]
        changed = [
            "ramp1_C_min",
            "hold1_T_C",
            "hold1_d_min",
            "ramp2_C_min",
            "hold2_T_C",
            "hold2_d_min",
        ]
        for column in fixed:
            if cycle[column].nunique(dropna=False) != 1:
                raise ValueError(
                    f"Cure-cycle comparison requires identical {column}."
                )
        if all(cycle[column].nunique(dropna=False) == 1 for column in changed):
            raise ValueError(
                "The two cure-cycle cases do not change any cure-cycle input."
            )

    thickness = cases[cases["study"] == "tool_thickness"]
    if not thickness.empty:
        fixed = [
            "ramp1_C_min", "hold1_T_C", "hold1_d_min",
            "ramp2_C_min", "hold2_T_C", "hold2_d_min",
            "hbot_W_m2K", "htop_W_m2K", "Lc_m",
            "total_process_time_s",
        ]
        for column in fixed:
            if thickness[column].nunique(dropna=False) != 1:
                raise ValueError(
                    f"Tool-thickness study requires identical {column}."
                )


def range_report(
    cases: pd.DataFrame,
    exp_config: dict,
    part_length_m: float,
) -> pd.DataFrame:
    rows = []
    for _, case in cases.iterrows():
        for column, (minimum_key, maximum_key) in RANGE_MAP.items():
            value = float(case[column])
            minimum = float(exp_config[minimum_key])
            maximum = float(exp_config[maximum_key])
            rows.append(
                {
                    "case_id": case["case_id"],
                    "parameter": column,
                    "value": value,
                    "training_min": minimum,
                    "training_max": maximum,
                    "inside_training_range": minimum <= value <= maximum,
                }
            )

        lc_value = float(case["Lc_m"])
        rows.append(
            {
                "case_id": case["case_id"],
                "parameter": "Lc_m",
                "value": lc_value,
                "training_min": part_length_m,
                "training_max": part_length_m,
                "inside_training_range": np.isclose(
                    lc_value,
                    part_length_m,
                    rtol=0.0,
                    atol=1e-12,
                ),
            }
        )

        total = float(case["total_process_time_s"])
        expected_total = float(exp_config["t_max"])
        rows.append(
            {
                "case_id": case["case_id"],
                "parameter": "total_process_time_s",
                "value": total,
                "training_min": expected_total,
                "training_max": expected_total,
                "inside_training_range": np.isclose(
                    total,
                    expected_total,
                    rtol=0.0,
                    atol=1e-9,
                ),
            }
        )
    return pd.DataFrame(rows)


def generate_random_cases(
    count: int,
    seed: int,
    margin_fraction: float,
    exp_config: dict,
    part_length_m: float,
) -> pd.DataFrame:
    if count < 1:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    if not 0.0 <= margin_fraction < 0.5:
        raise ValueError(
            "interior_margin_fraction must be in [0, 0.5)."
        )

    rng = np.random.default_rng(seed)
    rows = []

    for index in range(1, count + 1):
        row = {
            "study": "random_zero_shot",
            "case_id": f"random_{index:03d}",
            "Lc_m": part_length_m,
            "total_process_time_s": float(exp_config["t_max"]),
            "label": f"Random zero-shot case {index:03d}",
            "curve_label": f"Random {index:03d}",
        }

        for column, (minimum_key, maximum_key) in RANGE_MAP.items():
            minimum = float(exp_config[minimum_key])
            maximum = float(exp_config[maximum_key])
            span = maximum - minimum
            low = minimum + margin_fraction * span
            high = maximum - margin_fraction * span
            row[column] = float(rng.uniform(low, high))

        rows.append(row)

    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not XPIDON_ROOT.exists():
        raise FileNotFoundError(
            f"XPIDON repository not found: {XPIDON_ROOT}"
        )

    manual = pd.read_csv(MANUAL_CASES_FILE)
    validate_columns(manual)
    validate_study_rules(manual)

    settings = load_json(CONFIG_FILE)
    exp_config = load_json(XPIDON_ROOT / "exp_params.json")
    phy_config = load_json(XPIDON_ROOT / "phy_params.json")
    part_length_m = float(phy_config.get("part_len", 0.03))

    random_settings = settings.get("random_zero_shot", {})
    if bool(random_settings.get("enabled", True)):
        random_cases = generate_random_cases(
            count=int(random_settings.get("count", 20)),
            seed=int(random_settings.get("seed", 20260731)),
            margin_fraction=float(
                random_settings.get(
                    "interior_margin_fraction",
                    0.05,
                )
            ),
            exp_config=exp_config,
            part_length_m=part_length_m,
        )
    else:
        random_cases = pd.DataFrame(columns=REQUIRED_COLUMNS)

    runtime = pd.concat(
        [manual[REQUIRED_COLUMNS], random_cases],
        ignore_index=True,
    )
    validate_columns(runtime)

    report = range_report(runtime, exp_config, part_length_m)
    report.to_csv(RANGE_REPORT_FILE, index=False)

    outside = report[~report["inside_training_range"]]
    if not outside.empty:
        print(outside.to_string(index=False))
        raise ValueError(
            "At least one case is outside the trained XPIDON range. "
            f"See {RANGE_REPORT_FILE}."
        )

    runtime.to_csv(RUNTIME_CASES_FILE, index=False)

    print(
        f"Prepared {len(runtime)} cases: "
        f"{len(manual)} manual + {len(random_cases)} random."
    )
    print(f"Saved: {RUNTIME_CASES_FILE}")
    print(f"Saved: {RANGE_REPORT_FILE}")


if __name__ == "__main__":
    main()
