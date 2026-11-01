from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent

CASES_FILE = HERE / "cases.csv"

RESULT_FILE = (
    HERE
    / "outputs"
    / "fe_vs_xpidon_all_cases.csv"
)

METRICS_FILE = (
    HERE
    / "outputs"
    / "fe_vs_xpidon_metrics.csv"
)

FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


cases = pd.read_csv(
    CASES_FILE
)

results = pd.read_csv(
    RESULT_FILE
)

metrics = pd.read_csv(
    METRICS_FILE
)


# Select only the two cure-cycle cases
cycle_cases = cases[
    cases["study"] == "cure_cycle"
].copy()

if len(cycle_cases) != 2:
    raise RuntimeError(
        "Expected exactly two cases with study='cure_cycle'."
    )


fig, axes = plt.subplots(
    2,
    1,
    figsize=(9, 9),
    sharex=True,
)


for _, case in cycle_cases.iterrows():

    case_id = case["case_id"]
    curve_label = case["curve_label"]

    group = results[
        results["case_id"] == case_id
    ].sort_values("t_min")

    metric = metrics[
        metrics["case_id"] == case_id
    ].iloc[0]

    if group.empty:
        raise RuntimeError(
            f"No comparison results found for {case_id}"
        )

    # Temperature: FE
    axes[0].plot(
        group["t_min"],
        group["T_FE_C"],
        linewidth=2.0,
        label=f"FE — {curve_label}",
    )

    # Temperature: XPIDON
    axes[0].plot(
        group["t_min"],
        group["T_XPIDON_C"],
        linestyle="--",
        linewidth=1.8,
        label=f"XPIDON — {curve_label}",
    )

    # Cure-cycle air temperature
    axes[0].plot(
        group["t_min"],
        group["Ta_C"],
        linestyle=":",
        linewidth=1.5,
        label=f"Air — {curve_label}",
    )

    # Degree of cure: FE
    axes[1].plot(
        group["t_min"],
        group["alpha_FE"],
        linewidth=2.0,
        label=f"FE — {curve_label}",
    )

    # Degree of cure: XPIDON
    axes[1].plot(
        group["t_min"],
        group["alpha_XPIDON"],
        linestyle="--",
        linewidth=1.8,
        label=f"XPIDON — {curve_label}",
    )

    print(
        "\n",
        case_id,
        "\nTemperature RMSE:",
        metric["temperature_RMSE_C"],
        "\nTemperature max error:",
        metric["temperature_max_abs_error_C"],
        "\nFE peak temperature:",
        metric["FE_peak_temperature_C"],
        "\nXPIDON peak temperature:",
        metric["XPIDON_peak_temperature_C"],
        "\nPeak temperature error:",
        metric.get(
            "peak_temperature_error_C",
            metric["XPIDON_peak_temperature_C"]
            - metric["FE_peak_temperature_C"],
        ),
        "\nPeak-time error:",
        metric["peak_time_error_min"],
        "\nAlpha RMSE:",
        metric["alpha_RMSE"],
        "\nAlpha max error:",
        metric["alpha_max_abs_error"],
        "\nFE final alpha:",
        metric["FE_final_alpha"],
        "\nXPIDON final alpha:",
        metric["XPIDON_final_alpha"],
    )


axes[0].set_ylabel(
    "Temperature [°C]"
)

axes[0].set_title(
    "Different cure-cycle comparison at part midpoint"
)

axes[0].grid(
    True,
    linestyle="--",
    alpha=0.35,
)

axes[0].legend(
    fontsize=8,
)


axes[1].set_xlabel(
    "Time [min]"
)

axes[1].set_ylabel(
    "Degree of cure"
)

axes[1].grid(
    True,
    linestyle="--",
    alpha=0.35,
)

axes[1].legend(
    fontsize=8,
)


fig.tight_layout()

output_path = (
    FIG_DIR
    / "cure_cycle_comparison.png"
)

fig.savefig(
    output_path,
    dpi=300,
)

plt.close(fig)

print(
    "\nSaved:",
    output_path,
)