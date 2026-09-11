import argparse
import os
import sys
import json
import shutil
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, cohen_kappa_score, ConfusionMatrixDisplay

# Ensure the root of the project is in the Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from validation_sim.model_adapter import DRModelAdapter
from iqa_module.pipeline import process_retinal_image

REFERABLE_THRESHOLD = 2

def is_referable(grade):
    if pd.isna(grade): return np.nan
    return 1 if grade >= REFERABLE_THRESHOLD else 0

def calc_ci(p, n, z=1.96):
    """Calculate 95% CI using normal approximation (Wald interval)."""
    if n == 0: return None, None
    se = np.sqrt(p * (1 - p) / n)
    return max(0.0, p - z * se), min(1.0, p + z * se)

def calculate_metrics(df, subset_name):
    if len(df) == 0:
        return {
            "subset": subset_name,
            "count": 0,
            "sensitivity": None,
            "sensitivity_ci_lower": None,
            "sensitivity_ci_upper": None,
            "specificity": None,
            "specificity_ci_lower": None,
            "specificity_ci_upper": None,
            "qwk": None,
            "tp": 0, "tn": 0, "fp": 0, "fn": 0
        }

    y_true = df["true_grade"].values
    y_pred = df["predicted_grade"].values

    y_true_binary = np.array([is_referable(g) for g in y_true])
    y_pred_binary = np.array([is_referable(g) for g in y_pred])

    try:
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()
    except ValueError:
        tn, fp, fn, tp = 0, 0, 0, 0

    sens_denom = tp + fn
    spec_denom = tn + fp
    
    sensitivity = tp / sens_denom if sens_denom > 0 else np.nan
    specificity = tn / spec_denom if spec_denom > 0 else np.nan

    sens_ci_low, sens_ci_high = calc_ci(sensitivity, sens_denom) if not np.isnan(sensitivity) else (np.nan, np.nan)
    spec_ci_low, spec_ci_high = calc_ci(specificity, spec_denom) if not np.isnan(specificity) else (np.nan, np.nan)

    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=[0, 1, 2, 3, 4])

    return {
        "subset": subset_name,
        "count": len(df),
        "sensitivity": round(sensitivity, 4) if not np.isnan(sensitivity) else "N/A",
        "sensitivity_ci_lower": round(sens_ci_low, 4) if sens_ci_low is not None and not np.isnan(sens_ci_low) else "N/A",
        "sensitivity_ci_upper": round(sens_ci_high, 4) if sens_ci_high is not None and not np.isnan(sens_ci_high) else "N/A",
        "specificity": round(specificity, 4) if not np.isnan(specificity) else "N/A",
        "specificity_ci_lower": round(spec_ci_low, 4) if spec_ci_low is not None and not np.isnan(spec_ci_low) else "N/A",
        "specificity_ci_upper": round(spec_ci_high, 4) if spec_ci_high is not None and not np.isnan(spec_ci_high) else "N/A",
        "qwk": round(qwk, 4) if not np.isnan(qwk) else "N/A",
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }

def get_checkpoint_hash(checkpoint_path):
    import hashlib
    sha256_hash = hashlib.sha256()
    with open(checkpoint_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def plot_cm(df, title, filepath):
    if len(df) == 0: return
    cm = confusion_matrix(df["true_grade"], df["predicted_grade"], labels=[0, 1, 2, 3, 4])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3, 4])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.savefig(filepath)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate DR Model")
    parser.add_argument("--manifest", required=True, type=Path, help="Path to input manifest CSV")
    parser.add_argument("--dataset-root", required=True, type=Path, help="Root directory for images")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to model checkpoint")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for output files")
    parser.add_argument("--mode", required=True, choices=["baseline", "gated"], help="Evaluation mode")
    
    args = parser.parse_args()

    if not args.manifest.is_file():
        sys.exit(f"ERROR: Manifest not found at {args.manifest}")
    if not args.checkpoint.is_file():
        sys.exit(f"ERROR: Checkpoint not found at {args.checkpoint}")

    # Prepare output dir securely
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    try:
        df = pd.read_csv(args.manifest)
    except Exception as e:
        sys.exit(f"ERROR: Could not read manifest: {e}")

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt_hash = get_checkpoint_hash(args.checkpoint)
    try:
        model_adapter = DRModelAdapter()
        print("Model loaded successfully.")
    except Exception as e:
        sys.exit(f"ERROR loading model: {e}")

    results = []
    
    counts = {
        "manifest_total": len(df),
        "attempted": 0,
        "valid_predictions": 0,
        "inference_failures": 0,
        "iqa_pass": 0,
        "iqa_reject": 0,
        "iqa_error": 0
    }

    for index, row in df.iterrows():
        sample_id = row.get("sample_id", "unknown")
        rel_path = row["relative_path"]
        image_path = args.dataset_root / rel_path
        
        counts["attempted"] += 1
        
        iqa_status = "NOT_RUN"
        iqa_score = np.nan
        iqa_error = ""
        predicted_grade = np.nan
        raw_confidence = np.nan
        probs = []
        inference_status = "NOT_RUN"
        inference_error = ""
        
        # 1. Run IQA
        try:
            iqa_result = process_retinal_image(str(image_path))
            iqa_status = iqa_result["status"]
            iqa_score = iqa_result.get("quality_score", np.nan)
            if iqa_status == "PASS": counts["iqa_pass"] += 1
            elif iqa_status == "REJECT": counts["iqa_reject"] += 1
        except Exception as e:
            iqa_status = "ERROR"
            iqa_error = str(e)
            counts["iqa_error"] += 1

        # 2. Classifier Inference
        run_inference = True
        if args.mode == "gated" and iqa_status in ["REJECT", "ERROR"]:
            run_inference = False
            inference_status = "SKIPPED_BY_GATE"

        if run_inference:
            try:
                prediction = model_adapter.predict(str(image_path))
                predicted_grade = prediction["severity_grade"]
                raw_confidence = prediction["raw_confidence"]
                probs = prediction["probabilities"]
                inference_status = "SUCCESS"
                counts["valid_predictions"] += 1
            except Exception as e:
                inference_status = "ERROR"
                inference_error = str(e)
                counts["inference_failures"] += 1

        results.append({
            "sample_id": sample_id,
            "dataset": row.get("dataset"),
            "split": row.get("split"),
            "true_grade": row["true_grade"],
            "predicted_grade": predicted_grade,
            "probabilities": probs,
            "raw_confidence": raw_confidence,
            "iqa_status": iqa_status,
            "iqa_score": iqa_score,
            "inference_status": inference_status,
            "iqa_error_msg": iqa_error,
            "inference_error_msg": inference_error,
            "checkpoint_hash": ckpt_hash
        })

    if counts["valid_predictions"] == 0:
        sys.exit("ERROR: No valid predictions were produced.")

    results_df = pd.DataFrame(results)
    predictions_out = args.output_dir / "prediction_audit.csv"
    results_df.to_csv(predictions_out, index=False)
    
    # Filter valid predictions
    valid_df = results_df[results_df["inference_status"] == "SUCCESS"].copy()
    
    # Calculate metrics
    metrics_list = []
    
    # 1. All valid
    metrics_list.append(calculate_metrics(valid_df, "All Successfully Predicted External Images"))
    
    # 2. IQA Accepted
    valid_accepted = valid_df[valid_df["iqa_status"] == "PASS"]
    counts["valid_accepted"] = len(valid_accepted)
    metrics_list.append(calculate_metrics(valid_accepted, "IQA Accepted Images (Valid Predictions)"))
    
    # 3. IQA Rejected (Offline Analysis)
    valid_rejected = valid_df[valid_df["iqa_status"] == "REJECT"]
    metrics_list.append(calculate_metrics(valid_rejected, "IQA Rejected Images (Valid Predictions, Offline)"))

    metrics_df = pd.DataFrame(metrics_list)
    metrics_out = args.output_dir / "evaluation_metrics.csv"
    metrics_df.to_csv(metrics_out, index=False)

    # Confusion matrices
    plot_cm(valid_df, "Confusion Matrix (All Valid Predictions)", args.output_dir / "cm_all.png")
    plot_cm(valid_accepted, "Confusion Matrix (IQA Accepted)", args.output_dir / "cm_accepted.png")

    # Generate Report
    successful_iqa = counts["iqa_pass"] + counts["iqa_reject"]
    rejection_rate = counts["iqa_reject"] / successful_iqa if successful_iqa > 0 else 0

    report = f"""# Evaluation Report

## Metadata
- **Run Date**: {datetime.now().isoformat()}
- **Mode**: {args.mode.upper()}
- **Checkpoint SHA256**: {ckpt_hash}

## Execution Counts
- **Manifest Total**: {counts['manifest_total']}
- **Images Attempted**: {counts['attempted']}
- **Valid Predictions**: {counts['valid_predictions']}
- **Inference Failures**: {counts['inference_failures']}
- **IQA PASS**: {counts['iqa_pass']}
- **IQA REJECT**: {counts['iqa_reject']}
- **IQA ERROR**: {counts['iqa_error']}
- **IQA Accepted with Valid Predictions**: {counts.get('valid_accepted', 0)}

## IQA Rejection Rate
Calculated among successful IQA assessments ({successful_iqa}): **{rejection_rate:.1%}**

## Metrics
Target: Referable DR (Severity Grade >= 2)
*Note: This proxy does not include a separate macular-edema assessment.*

### Confidence Intervals
95% Confidence Intervals are provided for Sensitivity and Specificity using a Normal approximation (Wald interval). 
*Limitation: These CIs are calculated per-image. If images are clustered by patient, these bounds may be artificially narrow.*
"""
    with open(args.output_dir / "report.md", "w") as f:
        f.write(report)

if __name__ == "__main__":
    main()
