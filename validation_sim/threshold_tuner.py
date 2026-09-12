import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import sys

# Add root to sys.path to import model_adapter
sys.path.append(str(Path(__file__).resolve().parent))
from model_adapter import DRModelAdapter

def main():
    manifest_path = Path("results/development_manifest.csv")
    df = pd.read_csv(manifest_path)
    
    adapter = DRModelAdapter()
    
    results = []
    
    print("Running inference on Development Set (413 images)...")
    for idx, row in df.iterrows():
        image_path = Path("../external_data") / row['relative_path']
        true_grade = row['true_grade']
        
        pred = adapter.predict(image_path)
        probs = pred.get('calibrated_probabilities', pred['probabilities'])
        
        referable_score = sum(probs[2:])
        is_referable = int(true_grade >= 2)
        
        results.append({
            "sample_id": row['sample_id'],
            "true_grade": true_grade,
            "is_referable": is_referable,
            "referable_score": referable_score
        })
        
    results_df = pd.DataFrame(results)
    
    y_true = results_df['is_referable'].values
    y_scores = results_df['referable_score'].values
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    # Plot ROC
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # Plot PR
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (PPV)')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig("results/development_curves.png")
    print("Saved curves to results/development_curves.png")
    
    # Find threshold
    # Rule: among thresholds achieving sensitivity >=90%, maximize specificity
    valid_idx = np.where(tpr >= 0.90)[0]
    
    if len(valid_idx) == 0:
        print("Warning: No threshold achieves Sensitivity >= 90%.")
        best_idx = np.argmax(tpr)
    else:
        # specificity = 1 - fpr
        # Maximize specificity means minimize fpr
        valid_fprs = fpr[valid_idx]
        best_valid_idx = np.argmin(valid_fprs)
        best_idx = valid_idx[best_valid_idx]
        
    best_threshold = float(thresholds[best_idx])
    best_sensitivity = tpr[best_idx]
    best_specificity = 1 - fpr[best_idx]
    
    # Compute other metrics for the chosen threshold
    y_pred = (y_scores >= best_threshold).astype(int)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    referral_rate = np.sum(y_pred) / len(y_pred)
    
    print(f"\n--- Threshold Tuning Results ---")
    print(f"Optimal Threshold: {best_threshold:.4f}")
    print(f"Sensitivity: {best_sensitivity:.2%}")
    print(f"Specificity: {best_specificity:.2%}")
    print(f"PPV: {ppv:.2%}")
    print(f"NPV: {npv:.2%}")
    print(f"Referral Rate: {referral_rate:.2%}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    with open(config_dir / "operating_threshold.json", "w") as f:
        json.dump({"referral_threshold": best_threshold}, f, indent=2)
        
    print("\nSaved threshold to config/operating_threshold.json")
    
if __name__ == "__main__":
    main()
