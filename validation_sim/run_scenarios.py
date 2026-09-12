import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
import pandas as pd
from validation_sim.simpy_model import load_config, run_simulation


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main():
    config = load_config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_runs = []
    all_patients = []

    for scenario in config["scenarios"]:
        for seed in config["random_seeds"]:
            summary, patients = run_simulation(
                config,
                scenario,
                seed=seed,
            )

            # Basic consistency checks.
            assert (
                summary["patients_arrived"]
                == summary["completed_screenings"]
                + summary["pending_at_close"]
            )

            assert (
                summary["referrals_completed"]
                <= summary["completed_screenings"]
            )

            all_runs.append(summary)

            for patient in patients:
                all_patients.append({
                    "scenario": scenario["name"],
                    "seed": seed,
                    **patient,
                })

            print(
                f"{scenario['name']} | seed={seed} | "
                f"arrived={summary['patients_arrived']} | "
                f"completed={summary['completed_screenings']} | "
                f"pending={summary['pending_at_close']}"
            )

    runs = pd.DataFrame(all_runs)
    patients = pd.DataFrame(all_patients)

    runs.to_csv(
        RESULTS_DIR / "scenario_runs.csv",
        index=False,
    )

    patients.to_csv(
        RESULTS_DIR / "patient_records.csv",
        index=False,
    )

    metrics = [
        "patients_arrived",
        "completed_screenings",
        "pending_at_close",
        "recapture_attempts_started",
        "manual_review_routes",
        "referrals_completed",
        "average_queue_wait_completed_minutes",
        "p95_queue_wait_completed_minutes",
        "average_total_time_completed_minutes",
        "camera_queue_at_close",
        "ai_queue_at_close",
        "review_queue_at_close",
    ]

    grouped = runs.groupby("scenario", sort=False)

    means = grouped[metrics].mean().add_suffix("_mean")
    deviations = grouped[metrics].std().add_suffix("_std")

    comparison = means.join(deviations)
    comparison.insert(0, "runs", grouped.size())

    comparison.round(3).to_csv(
        RESULTS_DIR / "scenario_results.csv"
    )

    print("\nSCENARIO COMPARISON\n")

    print(
        comparison[[
            "completed_screenings_mean",
            "pending_at_close_mean",
            "average_queue_wait_completed_minutes_mean",
        ]].round(2).to_string()
    )

    print(f"\nResults saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()