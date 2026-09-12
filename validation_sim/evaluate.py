import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    cohen_kappa_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REFERABLE_THRESHOLD = 2
GRADES = [0, 1, 2, 3, 4]

AUDIT_COLUMNS = [
    "sample_id",
    "dataset",
    "split",
    "true_grade",
    "predicted_grade",
    "probabilities",
    "raw_confidence",
    "calibrated_confidence",
    "calibrated_probabilities",
    "referable_score",
    "is_referable",
    "iqa_status",
    "iqa_score",
    "inference_status",
    "iqa_error_msg",
    "inference_error_msg",
    "checkpoint_hash",
]


def is_referable(grade):
    if pd.isna(grade):
        return np.nan
    return int(grade >= REFERABLE_THRESHOLD)


def calc_ci(p, n, z=1.96):
    """Wilson confidence interval for a binomial proportion."""
    if n <= 0 or not np.isfinite(p):
        return None, None

    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator

    margin = (
        z
        * np.sqrt(
            p * (1 - p) / n
            + z**2 / (4 * n**2)
        )
        / denominator
    )

    return max(0.0, center - margin), min(1.0, center + margin)


def metric_value(value):
    if value is None or not np.isfinite(value):
        return "N/A"
    return round(float(value), 4)


def validate_grades(values, column_name):
    numeric = pd.to_numeric(values, errors="coerce")
    array = numeric.to_numpy(dtype=float)

    if (
        not np.isfinite(array).all()
        or not np.isin(array, GRADES).all()
    ):
        raise ValueError(
            f"{column_name} must contain integer grades from 0 to 4."
        )

    return numeric.astype(int)


def calculate_metrics(df, subset_name):
    output = {
        "subset": subset_name,
        "count": len(df),
        "sensitivity": "N/A",
        "sensitivity_ci_lower": "N/A",
        "sensitivity_ci_upper": "N/A",
        "specificity": "N/A",
        "specificity_ci_lower": "N/A",
        "specificity_ci_upper": "N/A",
        "qwk": "N/A",
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
    }

    for grade in GRADES:
        output[f"true_grade_{grade}_count"] = 0

    if df.empty:
        return output

    y_true = validate_grades(df["true_grade"], "true_grade")
    y_pred = validate_grades(df["predicted_grade"], "predicted_grade")

    for grade in GRADES:
        output[f"true_grade_{grade}_count"] = int(
            (y_true == grade).sum()
        )

    true_binary = (y_true >= REFERABLE_THRESHOLD).astype(int)
    if "is_referable" in df.columns:
        pred_binary = df["is_referable"].astype(int).to_numpy()
    else:
        pred_binary = (y_pred >= REFERABLE_THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        true_binary,
        pred_binary,
        labels=[0, 1],
    ).ravel()

    tn, fp, fn, tp = map(int, (tn, fp, fn, tp))

    output.update({
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    })

    positive_count = tp + fn
    negative_count = tn + fp

    if positive_count:
        sensitivity = tp / positive_count
        low, high = calc_ci(sensitivity, positive_count)

        output["sensitivity"] = metric_value(sensitivity)
        output["sensitivity_ci_lower"] = metric_value(low)
        output["sensitivity_ci_upper"] = metric_value(high)

    if negative_count:
        specificity = tn / negative_count
        low, high = calc_ci(specificity, negative_count)

        output["specificity"] = metric_value(specificity)
        output["specificity_ci_lower"] = metric_value(low)
        output["specificity_ci_upper"] = metric_value(high)

    # Kappa is undefined when both arrays contain only the same grade.
    if np.unique(
        np.concatenate([y_true.to_numpy(), y_pred.to_numpy()])
    ).size > 1:
        qwk = cohen_kappa_score(
            y_true,
            y_pred,
            labels=GRADES,
            weights="quadratic",
        )
        output["qwk"] = metric_value(qwk)

    return output


def get_checkpoint_hash(checkpoint_path):
    if not Path(checkpoint_path).is_file():
        return "Not found"
    digest = hashlib.sha256()

    with Path(checkpoint_path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()

def get_git_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT).decode("utf-8").strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.STDOUT).decode("utf-8").strip()
        return {
            "commit_sha": commit,
            "dirty_working_tree": len(status) > 0
        }
    except Exception:
        return {"commit_sha": "unavailable", "dirty_working_tree": "unavailable"}

def get_package_versions():
    packages = ["tensorflow", "keras", "numpy", "pandas", "scikit-learn", "opencv-python"]
    versions = {}
    for pkg in packages:
        try:
            import importlib.metadata
            versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            versions[pkg] = "not_installed"
    return versions

def get_code_hashes():
    files_to_hash = [
        "validation_sim/evaluate.py",
        "validation_sim/model_adapter.py",
        "iqa_module/pipeline.py",
        "iqa_module/config.py",
        "iqa_module/quality_assessment.py",
        "iqa_module/enhancement.py"
    ]
    hashes = {}
    for f in files_to_hash:
        p = ROOT / f
        hashes[f] = get_checkpoint_hash(p) if p.is_file() else "Not found"
    return hashes

def save_json(path, data):
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, allow_nan=False)


def plot_cm(df, title, output_path):
    if df.empty:
        return

    matrix = confusion_matrix(
        df["true_grade"],
        df["predicted_grade"],
        labels=GRADES,
    )

    fig, axis = plt.subplots(figsize=(7, 6))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=GRADES,
    )

    display.plot(
        ax=axis,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    axis.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def read_manifest(path):
    df = pd.read_csv(path)

    required = {
        "sample_id",
        "relative_path",
        "true_grade",
        "dataset",
        "split",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Manifest missing columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Manifest is empty.")

    for column in ["sample_id", "relative_path", "dataset", "split"]:
        if df[column].isna().any():
            raise ValueError(f"Manifest has missing {column} values.")

        df[column] = df[column].astype(str).str.strip()

        if df[column].eq("").any():
            raise ValueError(f"Manifest has empty {column} values.")

    if df["sample_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate sample_id values.")

    if df["relative_path"].duplicated().any():
        raise ValueError("Manifest contains duplicate image paths.")

    df["true_grade"] = validate_grades(
        df["true_grade"],
        "true_grade",
    )

    return df


def resolve_image_path(dataset_root, relative_path):
    relative_path = Path(relative_path)

    if relative_path.is_absolute():
        raise ValueError("Manifest image path must be relative.")

    image_path = (dataset_root / relative_path).resolve()

    try:
        image_path.relative_to(dataset_root)
    except ValueError as error:
        raise ValueError(
            "Image path points outside the dataset root."
        ) from error

    if not image_path.is_file():
        raise FileNotFoundError(f"Image missing: {image_path}")

    return image_path


def validate_prediction(prediction):
    probabilities = np.asarray(
        prediction["probabilities"],
        dtype=float,
    )

    if (
        probabilities.shape != (5,)
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.isclose(probabilities.sum(), 1.0, atol=1e-3)
    ):
        raise ValueError("Invalid prediction probabilities.")

    raw_grade = float(prediction["severity_grade"])

    if not np.isfinite(raw_grade) or raw_grade not in GRADES:
        raise ValueError("Predicted grade must be an integer from 0 to 4.")

    grade = int(raw_grade)

    if grade != int(np.argmax(probabilities)):
        raise ValueError("Predicted grade does not match probability argmax.")

    confidence = float(prediction["raw_confidence"])

    if (
        not np.isfinite(confidence)
        or not np.isclose(
            confidence,
            probabilities[grade],
            atol=1e-5,
        )
    ):
        raise ValueError("Raw confidence does not match predicted class.")

    return grade, confidence, probabilities.tolist(), prediction.get("calibrated_confidence"), prediction.get("calibrated_probabilities"), prediction.get("referable_score"), prediction.get("is_referable")


def main():
    parser = argparse.ArgumentParser(
        description="Frozen DR classifier evaluation."
    )

    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["baseline", "gated"],
    )

    args = parser.parse_args()

    if not args.manifest.is_file():
        parser.error(f"Manifest not found: {args.manifest}")

    if not args.dataset_root.is_dir():
        parser.error(f"Dataset root not found: {args.dataset_root}")

    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")

    dataset_root = args.dataset_root.resolve()

    try:
        df = read_manifest(args.manifest)
    except Exception as error:
        parser.error(f"Invalid manifest: {error}")

    # Never delete or overwrite previous runs.
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    print(f"Results directory: {output_dir}")

    checkpoint_hash = get_checkpoint_hash(args.checkpoint)

    metadata = {
        "run_id": run_id,
        "run_date": datetime.now().astimezone().isoformat(),
        "mode": args.mode,
        "checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": get_checkpoint_hash(args.manifest),
        "referable_grade_threshold": REFERABLE_THRESHOLD,
        "class_mapping": "0, 1, 2, 3, 4 (argmax of model output)",
        "preprocessing": "Original image, 224x224 RGB, values 0–1",
        "calibration": "Fitted on IDRiD Training subset via calibrator.py (Temperature scaling)" if Path("config/calibration.json").exists() else "Unavailable",
        "dataset_independence": "Not independently verified",
        "thresholding": "Tuned threshold on IDRiD Training set" if Path("config/operating_threshold.json").exists() else "Argmax class >= 2",
        "environment": {
            "python_version": sys.version,
            "packages": get_package_versions()
        },
        "git": get_git_info(),
        "code_hashes": get_code_hashes()
    }

    save_json(output_dir / "run_metadata.json", metadata)

    # Lazy imports allow metric tests without loading TensorFlow.
    try:
        from validation_sim.model_adapter import DRModelAdapter
        from iqa_module.pipeline import process_retinal_image

        model_adapter = DRModelAdapter(
            model_path=args.checkpoint
        )

        print("Model loaded successfully.")

    except Exception as error:
        save_json(
            output_dir / "run_error.json",
            {"stage": "initialization", "error": str(error)},
        )
        raise SystemExit(f"Initialization failed: {error}")

    counts = {
        "manifest_total": len(df),
        "attempted": 0,
        "input_failures": 0,
        "valid_predictions": 0,
        "inference_attempted": 0,
        "inference_failures": 0,
        "skipped_by_gate": 0,
        "iqa_pass": 0,
        "iqa_reject": 0,
        "iqa_error": 0,
        "iqa_not_run": 0,
        "valid_accepted": 0,
    }

    records = []

    for index, row in df.iterrows():
        counts["attempted"] += 1

        record = {
            "sample_id": row["sample_id"],
            "dataset": row["dataset"],
            "split": row["split"],
            "true_grade": int(row["true_grade"]),
            "predicted_grade": np.nan,
            "probabilities": "[]",
            "raw_confidence": np.nan,
            "calibrated_confidence": np.nan,
            "calibrated_probabilities": "[]",
            "referable_score": np.nan,
            "is_referable": np.nan,
            "iqa_status": "NOT_RUN",
            "iqa_score": np.nan,
            "inference_status": "NOT_RUN",
            "iqa_error_msg": "",
            "inference_error_msg": "",
            "checkpoint_hash": checkpoint_hash,
        }

        try:
            image_path = resolve_image_path(
                dataset_root,
                row["relative_path"],
            )

            # Check that image bytes still match the manifest.
            expected_hash = row.get("sha256")

            if pd.notna(expected_hash):
                actual_hash = get_checkpoint_hash(image_path)

                if actual_hash != str(expected_hash).strip().lower():
                    raise ValueError("Image SHA-256 differs from manifest.")

        except Exception as error:
            counts["input_failures"] += 1
            counts["iqa_not_run"] += 1

            record["inference_status"] = "INPUT_ERROR"
            record["inference_error_msg"] = str(error)

            records.append(record)
            continue

        try:
            iqa_result = process_retinal_image(str(image_path))

            status = iqa_result["status"]

            if status not in {"PASS", "REJECT"}:
                raise ValueError(f"Unexpected IQA status: {status}")

            score = float(iqa_result["quality_score"])

            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid IQA quality score.")

            record["iqa_status"] = status
            record["iqa_score"] = score

            if status == "PASS":
                counts["iqa_pass"] += 1
            else:
                counts["iqa_reject"] += 1

        except Exception as error:
            record["iqa_status"] = "ERROR"
            record["iqa_error_msg"] = str(error)
            counts["iqa_error"] += 1

        # Gated mode grades only explicitly accepted images.
        if args.mode == "gated" and record["iqa_status"] != "PASS":
            record["inference_status"] = "SKIPPED_BY_GATE"
            counts["skipped_by_gate"] += 1

        else:
            counts["inference_attempted"] += 1

            try:
                prediction = model_adapter.predict(str(image_path))

                grade, confidence, probabilities, cal_conf, cal_probs, ref_score, is_ref = validate_prediction(
                    prediction
                )

                record["predicted_grade"] = grade
                record["raw_confidence"] = confidence
                record["probabilities"] = json.dumps(probabilities)
                record["calibrated_confidence"] = cal_conf
                record["calibrated_probabilities"] = json.dumps(cal_probs) if cal_probs else "[]"
                record["referable_score"] = ref_score
                record["is_referable"] = is_ref
                record["inference_status"] = "SUCCESS"

                counts["valid_predictions"] += 1

                if record["iqa_status"] == "PASS":
                    counts["valid_accepted"] += 1

            except Exception as error:
                record["inference_status"] = "ERROR"
                record["inference_error_msg"] = str(error)
                counts["inference_failures"] += 1

        records.append(record)

        if (index + 1) % 20 == 0:
            print(f"Processed {index + 1}/{len(df)} images")

    # Persist every attempted record, including failures and abstentions.
    results_df = pd.DataFrame(records, columns=AUDIT_COLUMNS)

    results_df.to_csv(
        output_dir / "prediction_audit.csv",
        index=False,
    )

    save_json(
        output_dir / "execution_counts.json",
        counts,
    )

    if counts["valid_predictions"] == 0:
        message = (
            "No valid predictions were produced. "
            "All attempted records and execution counts were saved."
        )

        (output_dir / "report.md").write_text(
            f"# Evaluation did not produce metrics\n\n{message}\n",
            encoding="utf-8",
        )

        raise SystemExit(message)

    valid_df = results_df[
        results_df["inference_status"] == "SUCCESS"
    ].copy()

    accepted_df = valid_df[
        valid_df["iqa_status"] == "PASS"
    ].copy()

    rejected_df = valid_df[
        valid_df["iqa_status"] == "REJECT"
    ].copy()

    metrics = [
        calculate_metrics(
            valid_df,
            "All Successfully Predicted External Images",
        ),
        calculate_metrics(
            accepted_df,
            "IQA Accepted Images (Valid Predictions)",
        ),
        calculate_metrics(
            rejected_df,
            "IQA Rejected Images (Offline Analysis Only)",
        ),
    ]

    pd.DataFrame(metrics).to_csv(
        output_dir / "evaluation_metrics.csv",
        index=False,
    )

    plot_cm(
        valid_df,
        "All Successfully Predicted Images",
        output_dir / "cm_all.png",
    )

    plot_cm(
        accepted_df,
        "IQA Accepted Images",
        output_dir / "cm_accepted.png",
    )

    successful_iqa = counts["iqa_pass"] + counts["iqa_reject"]

    rejection_text = (
        f"{counts['iqa_reject'] / successful_iqa:.1%}"
        if successful_iqa
        else "N/A"
    )

    coverage_text = (
        f"{counts['valid_accepted'] / counts['attempted']:.1%}"
        if counts["attempted"]
        else "N/A"
    )

    count_lines = "\n".join(
        f"- **{name}**: {value}"
        for name, value in counts.items()
    )

    metric_rows = "\n".join(
        f"| {item['subset']} | {item['count']} | "
        f"{item['sensitivity']} | {item['specificity']} | "
        f"{item['qwk']} |"
        for item in metrics
    )

    report = f"""# Cross-Dataset Evaluation Report

## Run Information

- Run ID: {run_id}
- Mode: {args.mode}
- Checkpoint SHA-256: {checkpoint_hash}
- Dataset independence: not independently verified.

## Execution Counts

{count_lines}

## Coverage and Rejections

- IQA rejection rate among {successful_iqa} successful IQA assessments:
  {rejection_text}
- IQA accepted images with valid predictions / all attempted:
  {coverage_text}

## Metrics

Binary target: severity grade >= {REFERABLE_THRESHOLD}.
This is a grade-based referable-DR proxy without a separate
macular-edema assessment.

| Subset | Count | Sensitivity | Specificity | Weighted Kappa |
|---|---:|---:|---:|---:|
{metric_rows}

## Interpretation

- Baseline mode predicts on original images and groups results by IQA.
- Rejected-image predictions in baseline mode are offline analysis only.
- Gated mode skips inference unless IQA explicitly returns PASS.
- Accepted images use the original-image classifier preprocessing.
  This run does not evaluate enhanced-image classifier input.
- Raw confidence is not accuracy and is not calibrated.
- Sensitivity must be considered alongside specificity and coverage.
- Accepted-subset metrics alone do not prove that IQA improves performance.

## Confidence Intervals

The metrics CSV includes 95% Wilson score intervals for sensitivity
and specificity. Intervals are image-level and assume independent
observations. Patient clustering, if present, is not accounted for.

## Limitations

The checkpoint is reported as APTOS-trained. Exact training overlap
has not been independently verified. IQA thresholds were developed
using IDRiD data; full-pipeline independence remains unverified.

This is retrospective dataset evaluation, not proof of clinical
validation or deployment readiness.
"""

    (output_dir / "report.md").write_text(
        report,
        encoding="utf-8",
    )

    print("\nEvaluation complete.")
    print(pd.DataFrame(metrics)[
        ["subset", "count", "sensitivity", "specificity", "qwk"]
    ].to_string(index=False))
    print(f"\nResults saved in: {output_dir}")


if __name__ == "__main__":
    main()