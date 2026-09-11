import os
import sys
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

RESULTS_DIR = ROOT / "validation_sim" / "results"
MANIFEST_PATH = RESULTS_DIR / "test_manifest.csv"
PREDICTIONS_PATH = RESULTS_DIR / "prediction_results.csv"
METRICS_PATH = RESULTS_DIR / "cross_dataset_metrics.csv"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
EVALUATION_SUMMARY_PATH = RESULTS_DIR / "evaluation_summary.md"

# Define Referable DR threshold
# Grade 0: No DR, Grade 1: Mild DR -> Non-referable (0)
# Grade 2: Moderate DR, Grade 3: Severe DR, Grade 4: Proliferative DR -> Referable (1)
REFERABLE_THRESHOLD = 2

def is_referable(grade):
    return 1 if grade >= REFERABLE_THRESHOLD else 0

def calculate_metrics(df, subset_name):
    if len(df) == 0:
        return {
            "subset": subset_name,
            "count": 0,
            "sensitivity": None,
            "specificity": None,
            "qwk": None
        }

    y_true = df["true_grade"].values
    y_pred = df["predicted_grade"].values

    y_true_binary = np.array([is_referable(g) for g in y_true])
    y_pred_binary = np.array([is_referable(g) for g in y_pred])

    # Confusion Matrix for binary referable vs non-referable
    try:
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()
    except ValueError:
        tn, fp, fn, tp = 0, 0, 0, 0

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Quadratic Weighted Kappa
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=[0, 1, 2, 3, 4])

    return {
        "subset": subset_name,
        "count": len(df),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "qwk": round(qwk, 4)
    }

def main():
    print("Starting Cross-Dataset Evaluation...")
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    df = pd.read_csv(MANIFEST_PATH)
    print(f"Loaded manifest with {len(df)} images.")

    try:
        model_adapter = DRModelAdapter()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    results = []

    # Run Pipeline
    for index, row in df.iterrows():
        image_path = row["image_path"]
        
        # 1. Run IQA
        try:
            iqa_result = process_retinal_image(image_path)
            iqa_status = iqa_result["status"]
            iqa_score = iqa_result.get("quality_score", 0.0)
        except Exception as e:
            print(f"Error in IQA for {row['image_name']}: {e}")
            iqa_status = "ERROR"
            iqa_score = 0.0
            
        # 2. Run Model Inference
        # Requirements: "frozen model" expects raw image (model_adapter handles resize/norm).
        try:
            prediction = model_adapter.predict(image_path)
            predicted_grade = prediction["severity_grade"]
            raw_confidence = prediction["raw_confidence"]
            probs = prediction["probabilities"]
        except Exception as e:
            print(f"Error in Model for {row['image_name']}: {e}")
            predicted_grade = -1
            raw_confidence = 0.0
            probs = []

        results.append({
            "image_name": row["image_name"],
            "true_grade": row["true_grade"],
            "predicted_grade": predicted_grade,
            "iqa_status": iqa_status,
            "iqa_score": iqa_score,
            "raw_confidence": raw_confidence,
            "probabilities": probs
        })

        if (index + 1) % 20 == 0:
            print(f"Processed {index + 1}/{len(df)} images...")

    results_df = pd.DataFrame(results)
    
    # Filter out inference errors if any
    valid_results = results_df[results_df["predicted_grade"] >= 0]
    valid_results.to_csv(PREDICTIONS_PATH, index=False)
    print(f"\nSaved granular predictions to {PREDICTIONS_PATH}")

    # Metrics Calculation
    subset_all = valid_results
    subset_accepted = valid_results[valid_results["iqa_status"] == "PASS"]
    subset_rejected = valid_results[valid_results["iqa_status"] == "REJECT"]

    rejection_rate = len(subset_rejected) / len(valid_results) if len(valid_results) > 0 else 0

    metrics_list = [
        calculate_metrics(subset_all, "All External Images"),
        calculate_metrics(subset_accepted, "IQA Accepted Images"),
        calculate_metrics(subset_rejected, "IQA Rejected Images")
    ]
    
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df["rejection_rate"] = [rejection_rate, None, None]
    metrics_df.to_csv(METRICS_PATH, index=False)
    print(f"Saved metrics to {METRICS_PATH}")

    # Confusion Matrix (IQA Accepted subset)
    if len(subset_accepted) > 0:
        cm = confusion_matrix(
            subset_accepted["true_grade"], 
            subset_accepted["predicted_grade"], 
            labels=[0, 1, 2, 3, 4]
        )
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3, 4])
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix (IQA Accepted Images)")
        plt.savefig(CONFUSION_MATRIX_PATH)
        plt.close()
        print(f"Saved confusion matrix plot to {CONFUSION_MATRIX_PATH}")

    # Generate Markdown Summary
    summary_md = f"""# Cross-Dataset Evaluation Summary

## Dataset Information
- **Dataset**: IDRiD Testing Set (Untouched for 0% data leakage)
- **Total Images**: {len(valid_results)}
- **Referable DR Threshold**: Grade >= {REFERABLE_THRESHOLD} (Moderate, Severe, Proliferative)

## Pipeline Workflow
Images were evaluated using the IQA Module for quality triage, followed by the frozen APTOS-trained model.

## IQA Rejection Rate
- **Accepted**: {len(subset_accepted)}
- **Rejected**: {len(subset_rejected)}
- **Rejection Rate**: {rejection_rate:.1%}

## Metrics Summary

| Subset | Count | Sensitivity (Referable) | Specificity (Non-Referable) | Quadratic Weighted Kappa |
|---|---|---|---|---|
| All External Images | {metrics_list[0]['count']} | {metrics_list[0]['sensitivity']} | {metrics_list[0]['specificity']} | {metrics_list[0]['qwk']} |
| IQA Accepted Images | {metrics_list[1]['count']} | {metrics_list[1]['sensitivity']} | {metrics_list[1]['specificity']} | {metrics_list[1]['qwk']} |
| IQA Rejected Images | {metrics_list[2]['count']} | {metrics_list[2]['sensitivity']} | {metrics_list[2]['specificity']} | {metrics_list[2]['qwk']} |

## Observations
- This evaluation correctly isolates model performance based on image quality as determined by the IQA pipeline.
- Model raw confidence is recorded in the granular predictions but is explicitly **not** reported as accuracy.
- The confusion matrix has been generated specifically for the clinically relevant workflow (IQA Accepted images).
"""
    with open(EVALUATION_SUMMARY_PATH, "w") as f:
        f.write(summary_md)
    print(f"Saved evaluation summary to {EVALUATION_SUMMARY_PATH}")

if __name__ == "__main__":
    main()
