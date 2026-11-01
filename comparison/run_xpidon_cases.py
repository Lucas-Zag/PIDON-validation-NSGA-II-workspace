from __future__ import annotations

from pathlib import Path
import glob
import json
import pickle
import re
import sys

import jax.numpy as jnp
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
XPIDON_ROOT = WORKSPACE / "XPIDON-Composites-FV1"

if not XPIDON_ROOT.exists():
    raise FileNotFoundError(
        f"XPIDON repository not found: {XPIDON_ROOT}"
    )

sys.path.insert(0, str(XPIDON_ROOT))

from utils.phy_params import PhyParams
from utils.exp_params import ExpParams
from models.pidon import XPIDON


CASES_FILE = HERE / "outputs" / "cases_runtime.csv"
FE_FILE = HERE / "outputs" / "fe_key_curves.csv"
CONFIG_FILE = HERE / "comparison_config.json"
OUT_DIR = HERE / "outputs"
MODEL_DIR = XPIDON_ROOT / "outputs" / "models"

BRANCH = [9, 50, 50, 50, 100]
TRUNK = [2, 50, 50, 50, 50, 50, 100]
NOMAD = [100, 50, 50, 50, 50, 1]

POSITION_SPECS = {
    "tool_bottom": {
        "region": "tool",
        "x_norm": 0.0,
        "predictor": "tool",
    },
    "tool_mid": {
        "region": "tool",
        "x_norm": 0.5,
        "predictor": "tool",
    },
    "tool_interface": {
        "region": "tool",
        "x_norm": 1.0,
        "predictor": "tool",
    },
    "part_interface": {
        "region": "part",
        "x_norm": 0.0,
        "predictor": "part",
    },
    "part_mid": {
        "region": "part",
        "x_norm": 0.5,
        "predictor": "part",
    },
    "part_top": {
        "region": "part",
        "x_norm": 1.0,
        "predictor": "part",
    },
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_tag(path: str | Path) -> str:
    match = re.search(
        r"xpidon_class_(\d+)_(T|a)\.pkl",
        Path(path).name,
    )
    if not match:
        raise ValueError(f"Invalid checkpoint name: {path}")
    return match.group(1)


def tag_fraction(tag: str) -> float:
    return int(tag) / 10000.0


def scale_case(case: pd.Series, exp: ExpParams) -> np.ndarray:
    return np.array(
        [
            (float(case.ramp1_C_min) / 60.0)
            * (exp.T_scaler * exp.t_max),
            float(case.hold1_T_C) * exp.T_scaler,
            float(case.hold1_d_min) * 60.0 / exp.t_max,
            (float(case.ramp2_C_min) / 60.0)
            * (exp.T_scaler * exp.t_max),
            float(case.hold2_T_C) * exp.T_scaler,
            float(case.hold2_d_min) * 60.0 / exp.t_max,
            float(case.hbot_W_m2K) * exp.h_scaler,
            float(case.htop_W_m2K) * exp.h_scaler,
            float(case.Lt_m) * exp.len_scaler,
        ],
        dtype=np.float32,
    )


def build_subdomains(
    phy: PhyParams,
    exp: ExpParams,
):
    temperature_files = glob.glob(
        str(MODEL_DIR / "xpidon_class_*_T.pkl")
    )
    alpha_files = glob.glob(
        str(MODEL_DIR / "xpidon_class_*_a.pkl")
    )

    temperature_by_tag = {
        extract_tag(path): path
        for path in temperature_files
    }
    alpha_by_tag = {
        extract_tag(path): path
        for path in alpha_files
    }

    tags = sorted(
        set(temperature_by_tag) & set(alpha_by_tag),
        key=int,
    )

    if not tags:
        raise RuntimeError(
            f"No matched XPIDON checkpoints found in {MODEL_DIR}."
        )

    if tag_fraction(tags[-1]) < 0.9999:
        raise RuntimeError(
            "The checkpoint set does not cover global normalized time 1.0."
        )

    subdomains = []
    start_fraction = 0.0

    for tag in tags:
        end_fraction = tag_fraction(tag)
        if end_fraction <= start_fraction:
            raise RuntimeError(
                f"Invalid checkpoint order at tag {tag}."
            )

        model = XPIDON(
            phy,
            exp,
            BRANCH,
            TRUNK,
            NOMAD,
            start_fraction * exp.t_max,
            end_fraction * exp.t_max,
        )

        with open(
            temperature_by_tag[tag],
            "rb",
        ) as handle:
            temperature_params = pickle.load(handle)

        with open(
            alpha_by_tag[tag],
            "rb",
        ) as handle:
            alpha_params = pickle.load(handle)

        subdomains.append(
            {
                "tag": tag,
                "start": start_fraction,
                "end": end_fraction,
                "model": model,
                "temperature_params": temperature_params,
                "alpha_params": alpha_params,
            }
        )
        start_fraction = end_fraction

    return subdomains


def main() -> None:
    if not CASES_FILE.exists():
        raise FileNotFoundError(
            f"{CASES_FILE} does not exist. Run prepare_cases.py first."
        )
    if not FE_FILE.exists():
        raise FileNotFoundError(
            f"{FE_FILE} does not exist. Run run_fe_cases.py first."
        )

    settings = load_json(CONFIG_FILE)
    position_names = settings.get(
        "prediction_positions",
        list(POSITION_SPECS),
    )

    unknown = set(position_names) - set(POSITION_SPECS)
    if unknown:
        raise ValueError(
            f"Unknown prediction positions: {sorted(unknown)}"
        )

    cases = pd.read_csv(CASES_FILE)
    fe = pd.read_csv(FE_FILE)

    phy = PhyParams(str(XPIDON_ROOT / "phy_params.json"))
    exp = ExpParams(str(XPIDON_ROOT / "exp_params.json"))

    if not np.allclose(
        cases["Lc_m"].to_numpy(dtype=float),
        float(phy.part_len),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "The trained XPIDON has fixed part thickness "
            f"Lc={phy.part_len} m. All comparison cases must match it."
        )

    subdomains = build_subdomains(phy, exp)
    print(
        "Using subdomain tags:",
        [item["tag"] for item in subdomains],
    )

    rows = []

    for case_number, (_, case) in enumerate(cases.iterrows(), start=1):
        print(
            f"XPIDON predicting [{case_number}/{len(cases)}]: "
            f"{case.case_id}"
        )

        physical_times_s = np.sort(
            fe.loc[
                fe["case_id"] == case.case_id,
                "t_s",
            ]
            .drop_duplicates()
            .to_numpy(dtype=float)
        )

        if len(physical_times_s) == 0:
            raise RuntimeError(
                f"No FE time points found for {case.case_id}."
            )

        global_time = (
            physical_times_s
            / float(case.total_process_time_s)
        )
        case_vector = scale_case(case, exp)

        predicted_point_count = 0

        for subdomain_index, subdomain in enumerate(subdomains):
            start_fraction = float(subdomain["start"])
            end_fraction = float(subdomain["end"])
            is_last = subdomain_index == len(subdomains) - 1

            if is_last:
                time_mask = (
                    (global_time >= start_fraction - 1e-12)
                    & (global_time <= end_fraction + 1e-12)
                )
            else:
                time_mask = (
                    (global_time >= start_fraction - 1e-12)
                    & (global_time < end_fraction - 1e-12)
                )

            if not np.any(time_mask):
                continue

            local_time = np.clip(
                (
                    global_time[time_mask]
                    - start_fraction
                )
                / (end_fraction - start_fraction),
                0.0,
                1.0,
            )
            selected_times_s = physical_times_s[time_mask]
            repeated_case = np.tile(
                case_vector,
                (len(local_time), 1),
            )

            model = subdomain["model"]
            temperature_params = subdomain["temperature_params"]
            alpha_params = subdomain["alpha_params"]
            tag = subdomain["tag"]

            for position_name in position_names:
                spec = POSITION_SPECS[position_name]
                spatial_coordinate = np.full_like(
                    local_time,
                    float(spec["x_norm"]),
                )
                query = np.column_stack(
                    [local_time, spatial_coordinate]
                )

                if spec["predictor"] == "tool":
                    temperature_scaled = model.pred_tool(
                        temperature_params,
                        jnp.asarray(repeated_case),
                        jnp.asarray(query),
                    )
                    alpha_prediction = np.full(
                        len(local_time),
                        np.nan,
                    )
                else:
                    temperature_scaled = model.pred_T(
                        temperature_params,
                        jnp.asarray(repeated_case),
                        jnp.asarray(query),
                    )
                    alpha_prediction = np.asarray(
                        model.pred_a(
                            alpha_params,
                            jnp.asarray(repeated_case),
                            jnp.asarray(query),
                        )
                    ).reshape(-1)

                temperature_c = (
                    np.asarray(temperature_scaled).reshape(-1)
                    / exp.T_scaler
                )

                for time_s, temperature, alpha in zip(
                    selected_times_s,
                    temperature_c,
                    alpha_prediction,
                ):
                    rows.append(
                        {
                            "case_id": case.case_id,
                            "position": position_name,
                            "region": spec["region"],
                            "x_norm": float(spec["x_norm"]),
                            "t_s": float(time_s),
                            "t_min": float(time_s / 60.0),
                            "T_XPIDON_C": float(temperature),
                            "alpha_XPIDON": (
                                float(alpha)
                                if np.isfinite(alpha)
                                else np.nan
                            ),
                            "subdomain_tag": tag,
                        }
                    )
                    predicted_point_count += 1

        expected_point_count = (
            len(physical_times_s)
            * len(position_names)
        )
        if predicted_point_count != expected_point_count:
            raise RuntimeError(
                f"{case.case_id}: generated {predicted_point_count} "
                f"prediction rows, expected {expected_point_count}. "
                "Check subdomain checkpoint coverage."
            )

    predictions = pd.DataFrame(rows).sort_values(
        ["case_id", "position", "t_s"]
    )

    if predictions.empty:
        raise RuntimeError("No XPIDON predictions were generated.")

    output_path = OUT_DIR / "xpidon_predictions.csv"
    predictions.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
