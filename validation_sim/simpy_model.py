import json
import random
from pathlib import Path
import numpy as np
import simpy

MODULE_DIR = Path(__file__).resolve().parent

def load_config():
    config_path = MODULE_DIR / "config" / "scenarios.json"
    with config_path.open(encoding="utf-8") as file:
        return json.load(file)

def validate_settings(settings, closing_time):
    if settings.get("patients_per_hour", 0) <= 0:
        raise ValueError("patients_per_hour must be positive")
    if settings.get("cameras", 0) <= 0 or not isinstance(settings["cameras"], int):
        raise ValueError("cameras capacity must be a positive integer")
    if settings.get("ai_workers", 0) <= 0 or not isinstance(settings["ai_workers"], int):
        raise ValueError("ai_workers capacity must be a positive integer")
    if settings.get("reviewers", 0) <= 0 or not isinstance(settings["reviewers"], int):
        raise ValueError("reviewers capacity must be a positive integer")
    if settings.get("max_capture_attempts", 0) <= 0 or not isinstance(settings["max_capture_attempts"], int):
        raise ValueError("max_capture_attempts must be a positive integer")
    
    if settings.get("capture_time_minutes", 0) <= 0:
        raise ValueError("capture_time_minutes must be positive")
    if settings.get("iqa_time_minutes", 0) <= 0:
        raise ValueError("iqa_time_minutes must be positive")
    if settings.get("ai_time_minutes", 0) <= 0:
        raise ValueError("ai_time_minutes must be positive")
    if settings.get("review_time_minutes", 0) <= 0:
        raise ValueError("review_time_minutes must be positive")
    
    if not (0 <= settings.get("iqa_rejection_probability", -1) <= 1):
        raise ValueError("iqa_rejection_probability must be in [0, 1]")
    if not (0 <= settings.get("referral_probability", -1) <= 1):
        raise ValueError("referral_probability must be in [0, 1]")
        
    if closing_time <= 0:
        raise ValueError("working_day_minutes must be positive")

def run_simulation(config, scenario, seed=42):
    """Run one screening day using assumed service times."""
    arrival_rng = random.Random(seed)
    # Use a separate RNG for outcomes so queue dynamics don't change arrival stream
    outcome_rng = random.Random(seed + 1)
    
    env = simpy.Environment()

    settings = {**config["defaults"], **scenario}
    closing_time = config.get("working_day_minutes", 480)

    validate_settings(settings, closing_time)

    cameras = simpy.Resource(env, capacity=settings["cameras"])
    ai_workers = simpy.Resource(env, capacity=settings["ai_workers"])
    reviewers = simpy.Resource(env, capacity=settings["reviewers"])

    patients = []

    def patient_journey(patient):
        passed_iqa = False

        for attempt in range(settings["max_capture_attempts"]):
            patient["stage"] = "camera_queue"
            queue_entered = env.now

            with cameras.request() as request:
                yield request
                patient["queue_wait_minutes"] += env.now - queue_entered

                patient["stage"] = "capture"
                patient["capture_attempts"] += 1
                yield env.timeout(settings["capture_time_minutes"])

                patient["stage"] = "iqa"
                yield env.timeout(settings["iqa_time_minutes"])

                # Draw rejection from pre-generated sequence for this patient
                rejected = patient["iqa_rejection_rolls"][attempt]

            if not rejected:
                passed_iqa = True
                break

            patient["iqa_rejections"] += 1

        patient["passed_iqa"] = passed_iqa

        if passed_iqa:
            patient["stage"] = "ai_queue"
            queue_entered = env.now

            with ai_workers.request() as request:
                yield request
                patient["queue_wait_minutes"] += env.now - queue_entered

                patient["stage"] = "ai_processing"
                yield env.timeout(settings["ai_time_minutes"])
        else:
            patient["manual_review_required"] = True

        patient["stage"] = "review_queue"
        queue_entered = env.now

        with reviewers.request() as request:
            yield request
            patient["queue_wait_minutes"] += env.now - queue_entered

            patient["stage"] = "doctor_review"
            yield env.timeout(settings["review_time_minutes"])

        patient["referred"] = patient["referral_roll"]
        patient["finished_at"] = env.now
        patient["total_time_minutes"] = env.now - patient["arrived_at"]
        patient["stage"] = "completed"

    def generate_arrivals():
        mean_gap_minutes = 60 / settings["patients_per_hour"]

        while True:
            gap = arrival_rng.expovariate(1 / mean_gap_minutes)
            yield env.timeout(gap)

            if env.now >= closing_time:
                return

            # Pre-generate outcomes to decouple RNGs
            iqa_rejection_rolls = [
                outcome_rng.random() < settings["iqa_rejection_probability"]
                for _ in range(settings["max_capture_attempts"])
            ]
            referral_roll = outcome_rng.random() < settings["referral_probability"]

            patient = {
                "patient_id": len(patients) + 1,
                "arrived_at": env.now,
                "stage": "arrived",
                "capture_attempts": 0,
                "iqa_rejections": 0,
                "passed_iqa": None,
                "manual_review_required": False,
                "queue_wait_minutes": 0.0,
                "finished_at": None,
                "total_time_minutes": None,
                "referred": None,
                "iqa_rejection_rolls": iqa_rejection_rolls,
                "referral_roll": referral_roll
            }

            patients.append(patient)
            env.process(patient_journey(patient))

    env.process(generate_arrivals())

    env.run(until=closing_time)

    completed = [
        patient for patient in patients
        if patient["stage"] == "completed"
    ]

    def completed_average(field):
        if not completed:
            return None
        return round(sum(patient[field] for patient in completed) / len(completed), 2)
        
    def completed_percentile(field, p=95):
        if not completed:
            return None
        return round(np.percentile([patient[field] for patient in completed], p), 2)

    summary = {
        "scenario": scenario["name"],
        "seed": seed,
        "patients_arrived": len(patients),
        "completed_screenings": len(completed),
        "pending_at_close": len(patients) - len(completed),
        "recapture_attempts_started": sum(
            max(0, patient["capture_attempts"] - 1)
            for patient in patients
        ),
        "manual_review_routes": sum(
            patient["manual_review_required"] for patient in patients
        ),
        "referrals_completed": sum(
            patient["referred"] is True for patient in completed
        ),
        "average_queue_wait_completed_minutes": completed_average("queue_wait_minutes"),
        "p95_queue_wait_completed_minutes": completed_percentile("queue_wait_minutes", 95),
        "average_total_time_completed_minutes": completed_average("total_time_minutes"),
        "camera_queue_at_close": len(cameras.queue),
        "ai_queue_at_close": len(ai_workers.queue),
        "review_queue_at_close": len(reviewers.queue),
    }

    return summary, patients

if __name__ == "__main__":
    config = load_config()
    baseline = config["scenarios"][0]

    summary, patients = run_simulation(
        config,
        baseline,
        seed=config["random_seeds"][0],
    )

    print("\nBASELINE SCREENING SIMULATION\n")
    print(json.dumps(summary, indent=2))