from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
FIG_DIR = HERE / "figures"
CONFIG_FILE = HERE / "comparison_config.json"

CASES_FILE = OUT_DIR / "cases_runtime.csv"
FE_FILE = OUT_DIR / "fe_key_curves.csv"
XPIDON_FILE = OUT_DIR / "xpidon_predictions.csv"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mae(values: pd.Series) -> float:
    array = values.to_numpy(dtype=float)
    return float(np.mean(np.abs(array)))


def rmse(values: pd.Series) -> float:
    array = values.to_numpy(dtype=float)
    return float(np.sqrt(np.mean(array**2)))


def calculate_metrics(group: pd.DataFrame) -> dict:
    group = group.sort_values("t_s").copy()

    fe_peak_row = group.loc[group["T_FE_C"].idxmax()]
    xp_peak_row = group.loc[group["T_XPIDON_C"].idxmax()]

    output = {
        "case_id": group["case_id"].iloc[0],
        "position": group["position"].iloc[0],
        "region": group["region"].iloc[0],
        "temperature_MAE_C": mae(group["T_error_C"]),
        "temperature_RMSE_C": rmse(group["T_error_C"]),
        "temperature_max_abs_error_C": float(
            np.max(np.abs(group["T_error_C"]))
        ),
        "FE_peak_temperature_C": float(
            fe_peak_row["T_FE_C"]
        ),
        "XPIDON_peak_temperature_C": float(
            xp_peak_row["T_XPIDON_C"]
        ),
        "peak_temperature_error_C": float(
            xp_peak_row["T_XPIDON_C"]
            - fe_peak_row["T_FE_C"]
        ),
        "FE_peak_time_min": float(
            fe_peak_row["t_min"]
        ),
        "XPIDON_peak_time_min": float(
            xp_peak_row["t_min"]
        ),
        "peak_time_error_min": float(
            xp_peak_row["t_min"]
            - fe_peak_row["t_min"]
        ),
        "alpha_MAE": np.nan,
        "alpha_RMSE": np.nan,
        "alpha_max_abs_error": np.nan,
        "FE_final_alpha": np.nan,
        "XPIDON_final_alpha": np.nan,
        "final_alpha_error": np.nan,
        "FE_max_cure_rate_time_min": np.nan,
        "XPIDON_max_cure_rate_time_min": np.nan,
        "max_cure_rate_time_error_min": np.nan,
    }

    valid_alpha = group[
        group["alpha_FE"].notna()
        & group["alpha_XPIDON"].notna()
    ].copy()

    if len(valid_alpha) >= 2:
        alpha_error = (
            valid_alpha["alpha_XPIDON"]
            - valid_alpha["alpha_FE"]
        )
        fe_rate = np.gradient(
            valid_alpha["alpha_FE"].to_numpy(dtype=float),
            valid_alpha["t_s"].to_numpy(dtype=float),
        )
        xp_rate = np.gradient(
            valid_alpha["alpha_XPIDON"].to_numpy(dtype=float),
            valid_alpha["t_s"].to_numpy(dtype=float),
        )
        fe_rate_index = int(np.argmax(fe_rate))
        xp_rate_index = int(np.argmax(xp_rate))

        output.update(
            {
                "alpha_MAE": mae(alpha_error),
                "alpha_RMSE": rmse(alpha_error),
                "alpha_max_abs_error": float(
                    np.max(np.abs(alpha_error))
                ),
                "FE_final_alpha": float(
                    valid_alpha["alpha_FE"].iloc[-1]
                ),
                "XPIDON_final_alpha": float(
                    valid_alpha["alpha_XPIDON"].iloc[-1]
                ),
                "final_alpha_error": float(
                    valid_alpha["alpha_XPIDON"].iloc[-1]
                    - valid_alpha["alpha_FE"].iloc[-1]
                ),
                "FE_max_cure_rate_time_min": float(
                    valid_alpha["t_min"].iloc[fe_rate_index]
                ),
                "XPIDON_max_cure_rate_time_min": float(
                    valid_alpha["t_min"].iloc[xp_rate_index]
                ),
                "max_cure_rate_time_error_min": float(
                    valid_alpha["t_min"].iloc[xp_rate_index]
                    - valid_alpha["t_min"].iloc[fe_rate_index]
                ),
            }
        )

    return output


def plot_study(
    study_name: str,
    study_cases: pd.DataFrame,
    merged: pd.DataFrame,
    part_mid_metrics: pd.DataFrame,
) -> None:
    if study_cases.empty:
        return

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.5, 8.5),
        sharex=True,
    )

    metric_lines = []

    for case_index, (_, case) in enumerate(
        study_cases.iterrows()
    ):
        case_id = case["case_id"]
        curve_label = case["curve_label"]
        group = merged[
            (merged["case_id"] == case_id)
            & (merged["position"] == "part_mid")
        ].sort_values("t_min")

        if group.empty:
            raise RuntimeError(
                f"No part_mid comparison result for {case_id}."
            )

        color = f"C{case_index}"

        axes[0].plot(
            group["t_min"],
            group["T_FE_C"],
            color=color,
            linewidth=2.2,
            label=f"FE — {curve_label}",
        )
        axes[0].plot(
            group["t_min"],
            group["T_XPIDON_C"],
            color=color,
            linestyle="--",
            linewidth=1.8,
            label=f"XPIDON — {curve_label}",
        )
        axes[0].plot(
            group["t_min"],
            group["Ta_C"],
            color=color,
            linestyle=":",
            linewidth=1.5,
            alpha=0.8,
            label=f"Cure cycle — {curve_label}",
        )

        axes[1].plot(
            group["t_min"],
            group["alpha_FE"],
            color=color,
            linewidth=2.2,
            label=f"FE — {curve_label}",
        )
        axes[1].plot(
            group["t_min"],
            group["alpha_XPIDON"],
            color=color,
            linestyle="--",
            linewidth=1.8,
            label=f"XPIDON — {curve_label}",
        )

        metric = part_mid_metrics[
            part_mid_metrics["case_id"] == case_id
        ].iloc[0]
        metric_lines.append(
            f"{curve_label}: "
            f"T RMSE={metric['temperature_RMSE_C']:.3f}°C, "
            f"T max={metric['temperature_max_abs_error_C']:.3f}°C, "
            f"peak Δt={metric['peak_time_error_min']:+.2f} min, "
            f"α RMSE={metric['alpha_RMSE']:.5f}"
        )

    axes[0].set_ylabel("Temperature [°C]")
    axes[0].set_title(
        f"{study_name.replace('_', ' ').title()} comparison "
        "at composite part midpoint"
    )
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].set_xlabel("Time [min]")
    axes[1].set_ylabel("Degree of cure")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(fontsize=8, ncol=2)

    fig.text(
        0.02,
        0.01,
        "\n".join(metric_lines),
        fontsize=8,
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    output_path = FIG_DIR / f"study_{study_name}.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_through_thickness(
    case_id: str,
    merged: pd.DataFrame,
) -> None:
    case_data = merged[
        merged["case_id"] == case_id
    ].copy()
    if case_data.empty:
        raise ValueError(
            f"through_thickness_case_id={case_id!r} was not found."
        )

    temperature_positions = [
        "tool_bottom",
        "tool_mid",
        "tool_interface",
        "part_interface",
        "part_mid",
        "part_top",
    ]
    alpha_positions = [
        "part_interface",
        "part_mid",
        "part_top",
    ]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 9),
        sharex=True,
    )

    for position_index, position in enumerate(
        temperature_positions
    ):
        group = case_data[
            case_data["position"] == position
        ].sort_values("t_min")
        if group.empty:
            continue
        color = f"C{position_index}"
        axes[0].plot(
            group["t_min"],
            group["T_FE_C"],
            color=color,
            linewidth=1.9,
            label=f"FE — {position}",
        )
        axes[0].plot(
            group["t_min"],
            group["T_XPIDON_C"],
            color=color,
            linestyle="--",
            linewidth=1.5,
            label=f"XPIDON — {position}",
        )

    for position_index, position in enumerate(alpha_positions):
        group = case_data[
            case_data["position"] == position
        ].sort_values("t_min")
        if group.empty:
            continue
        color = f"C{position_index}"
        axes[1].plot(
            group["t_min"],
            group["alpha_FE"],
            color=color,
            linewidth=1.9,
            label=f"FE — {position}",
        )
        axes[1].plot(
            group["t_min"],
            group["alpha_XPIDON"],
            color=color,
            linestyle="--",
            linewidth=1.5,
            label=f"XPIDON — {position}",
        )

    axes[0].set_title(
        f"Through-thickness validation: {case_id}"
    )
    axes[0].set_ylabel("Temperature [°C]")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(fontsize=7, ncol=2)

    axes[1].set_xlabel("Time [min]")
    axes[1].set_ylabel("Degree of cure")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(fontsize=8, ncol=2)

    fig.tight_layout()
    output_path = (
        FIG_DIR
        / f"through_thickness_{case_id}.png"
    )
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def random_summary(
    cases: pd.DataFrame,
    part_mid_metrics: pd.DataFrame,
) -> None:
    random_case_ids = cases.loc[
        cases["study"] == "random_zero_shot",
        "case_id",
    ].tolist()

    random_metrics = part_mid_metrics[
        part_mid_metrics["case_id"].isin(random_case_ids)
    ].copy()

    random_metrics.to_csv(
        OUT_DIR / "random_zero_shot_case_metrics.csv",
        index=False,
    )

    if random_metrics.empty:
        return

    temperature_worst_index = (
        random_metrics["temperature_RMSE_C"].idxmax()
    )
    alpha_worst_index = (
        random_metrics["alpha_RMSE"].idxmax()
    )
    final_alpha_abs = np.abs(
        random_metrics["final_alpha_error"]
    )
    peak_time_abs = np.abs(
        random_metrics["peak_time_error_min"]
    )

    summary = {
        "number_of_cases": len(random_metrics),
        "mean_temperature_RMSE_C": float(
            random_metrics["temperature_RMSE_C"].mean()
        ),
        "worst_temperature_RMSE_C": float(
            random_metrics["temperature_RMSE_C"].max()
        ),
        "worst_temperature_RMSE_case": random_metrics.loc[
            temperature_worst_index,
            "case_id",
        ],
        "mean_temperature_max_abs_error_C": float(
            random_metrics[
                "temperature_max_abs_error_C"
            ].mean()
        ),
        "worst_temperature_max_abs_error_C": float(
            random_metrics[
                "temperature_max_abs_error_C"
            ].max()
        ),
        "mean_alpha_RMSE": float(
            random_metrics["alpha_RMSE"].mean()
        ),
        "worst_alpha_RMSE": float(
            random_metrics["alpha_RMSE"].max()
        ),
        "worst_alpha_RMSE_case": random_metrics.loc[
            alpha_worst_index,
            "case_id",
        ],
        "mean_abs_final_alpha_error": float(
            final_alpha_abs.mean()
        ),
        "worst_abs_final_alpha_error": float(
            final_alpha_abs.max()
        ),
        "mean_abs_peak_time_error_min": float(
            peak_time_abs.mean()
        ),
        "worst_abs_peak_time_error_min": float(
            peak_time_abs.max()
        ),
    }

    summary_frame = pd.DataFrame([summary])
    summary_frame.to_csv(
        OUT_DIR / "random_zero_shot_summary.csv",
        index=False,
    )

    plot_data = random_metrics.sort_values("case_id")
    x = np.arange(len(plot_data))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=True,
    )

    axes[0].plot(
        x,
        plot_data["temperature_RMSE_C"],
        marker="o",
        label="Temperature RMSE [°C]",
    )
    axes[0].plot(
        x,
        plot_data["temperature_max_abs_error_C"],
        marker="s",
        label="Maximum temperature error [°C]",
    )
    axes[0].set_ylabel("Temperature error [°C]")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend()

    axes[1].plot(
        x,
        plot_data["alpha_RMSE"],
        marker="o",
        label="Degree-of-cure RMSE",
    )
    axes[1].plot(
        x,
        np.abs(plot_data["final_alpha_error"]),
        marker="s",
        label="Absolute final α error",
    )
    axes[1].set_xlabel("Random zero-shot case")
    axes[1].set_ylabel("Degree-of-cure error")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend()

    step = max(1, len(plot_data) // 10)
    tick_indices = x[::step]
    tick_labels = plot_data["case_id"].iloc[::step]
    axes[1].set_xticks(tick_indices)
    axes[1].set_xticklabels(
        tick_labels,
        rotation=45,
        ha="right",
    )

    fig.suptitle(
        "Random zero-shot validation across unseen cases"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_path = FIG_DIR / "random_zero_shot_validation.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print("\nRandom zero-shot aggregate summary:")
    print(summary_frame.to_string(index=False))
    print(f"Saved: {output_path}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    settings = load_json(CONFIG_FILE)
    cases = pd.read_csv(CASES_FILE)
    fe = pd.read_csv(FE_FILE)
    xp = pd.read_csv(XPIDON_FILE)

    fe_columns = [
        "case_id",
        "position",
        "region",
        "t_s",
        "t_min",
        "T_FE_C",
        "alpha_FE",
        "Ta_C",
    ]
    xp_columns = [
        "case_id",
        "position",
        "region",
        "x_norm",
        "t_s",
        "T_XPIDON_C",
        "alpha_XPIDON",
        "subdomain_tag",
    ]

    merged = fe[fe_columns].merge(
        xp[xp_columns],
        on=["case_id", "position", "region", "t_s"],
        how="inner",
        validate="one_to_one",
    )

    merged["T_error_C"] = (
        merged["T_XPIDON_C"]
        - merged["T_FE_C"]
    )
    merged["alpha_error"] = (
        merged["alpha_XPIDON"]
        - merged["alpha_FE"]
    )

    metrics = pd.DataFrame(
        [
            calculate_metrics(group)
            for _, group in merged.groupby(
                ["case_id", "position"],
                sort=False,
            )
        ]
    )

    part_mid_metrics = metrics[
        metrics["position"] == "part_mid"
    ].copy()

    merged.to_csv(
        OUT_DIR / "fe_vs_xpidon_all_cases.csv",
        index=False,
    )
    metrics.to_csv(
        OUT_DIR / "fe_vs_xpidon_metrics_by_position.csv",
        index=False,
    )
    part_mid_metrics.to_csv(
        OUT_DIR / "fe_vs_xpidon_metrics.csv",
        index=False,
    )

    manual_studies = cases[
        cases["study"] != "random_zero_shot"
    ]["study"].drop_duplicates()

    for study_name in manual_studies:
        plot_study(
            study_name=study_name,
            study_cases=cases[cases["study"] == study_name],
            merged=merged,
            part_mid_metrics=part_mid_metrics,
        )

    through_case = settings.get(
        "through_thickness_case_id",
        "cure_cycle_1",
    )
    plot_through_thickness(
        case_id=through_case,
        merged=merged,
    )

    random_summary(
        cases=cases,
        part_mid_metrics=part_mid_metrics,
    )

    print("\nPart-midpoint metrics:")
    print(part_mid_metrics.to_string(index=False))
    print("\nSaved comparison outputs and figures.")


if __name__ == "__main__":
    main()
