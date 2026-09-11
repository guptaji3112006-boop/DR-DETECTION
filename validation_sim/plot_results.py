from pathlib import Path

import matplotlib

# Save charts without opening a separate desktop window.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main():
    csv_path = RESULTS_DIR / "scenario_results.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            "scenario_results.csv missing. Run run_scenarios first."
        )

    data = pd.read_csv(csv_path)

    labels = (
        data["scenario"]
        .str.replace("_", " ", regex=False)
        .str.title()
    )

    charts = [
        (
            "completed_screenings",
            "Completed Screenings",
            "Patients",
        ),
        (
            "pending_at_close",
            "Pending Patients at Closing",
            "Patients",
        ),
        (
            "average_queue_wait_completed_minutes",
            "Queue Wait — Completed Patients Only",
            "Minutes",
        ),
        (
            "recapture_attempts_started",
            "Recapture Attempts Started",
            "Attempts",
        ),
    ]

    colors = ["#2563eb", "#f59e0b", "#10b981"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for axis, (metric, title, unit) in zip(axes.flat, charts):
        means = data[f"{metric}_mean"]
        stds = data[f"{metric}_std"].fillna(0)

        bars = axis.bar(
            labels,
            means,
            yerr=stds,
            capsize=5,
            color=colors,
            width=0.55,
        )

        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.set_ylabel(unit)
        axis.set_ylim(bottom=0)
        axis.margins(y=0.2)

        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)

        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

        axis.tick_params(axis="x", labelsize=9)

        for bar, mean, std in zip(bars, means, stds):
            if pd.notna(mean):
                axis.annotate(
                    f"{mean:.1f}",
                    xy=(
                        bar.get_x() + bar.get_width() / 2,
                        mean + std,
                    ),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=10,
                )

    fig.suptitle(
        "District Screening Workflow — Scenario Comparison",
        fontsize=17,
        fontweight="bold",
    )

    fig.text(
        0.5,
        0.025,
        "Illustrative simulation assumptions. Error bars: standard "
        "deviation across runs.\n"
        "Waiting-time averages exclude unfinished patients; "
        "interpret alongside the closing backlog.",
        ha="center",
        fontsize=10,
        color="#475569",
    )

    fig.tight_layout(rect=(0, 0.08, 1, 0.94))

    output_path = RESULTS_DIR / "scenario_comparison.png"

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print("\nCharts generated successfully!")
    print(f"Saved at: {output_path}")


if __name__ == "__main__":
    main()