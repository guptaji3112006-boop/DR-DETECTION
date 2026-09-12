import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
import sys

# Add root to sys.path to import model_adapter
sys.path.append(str(Path(__file__).resolve().parent))
from model_adapter import DRModelAdapter

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece

def main():
    manifest_path = Path("results/development_manifest.csv")
    df = pd.read_csv(manifest_path)
    
    adapter = DRModelAdapter()
    
    y_true_list = []
    prob_list = []
    
    print("Extracting probabilities on Development Set (413 images) for Calibration...")
    for idx, row in df.iterrows():
        image_path = Path("../external_data") / row['relative_path']
        true_grade = row['true_grade']
        
        pred = adapter.predict(image_path)
        probs = pred['probabilities']
        
        y_true_list.append(true_grade)
        prob_list.append(probs)
        
    y_true = np.array(y_true_list)
    probs_np = np.array(prob_list)
    
    # Avoid log(0)
    eps = 1e-7
    probs_np = np.clip(probs_np, eps, 1.0 - eps)
    
    # Recover logits (up to a constant shift)
    logits = np.log(probs_np)
    
    # NLL function for temperature T
    def nll_with_temperature(T):
        scaled_logits = logits / T
        # softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        scaled_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        scaled_probs = np.clip(scaled_probs, eps, 1.0 - eps)
        
        # NLL
        loss = -np.mean(np.log(scaled_probs[np.arange(len(y_true)), y_true]))
        return loss

    initial_T = 1.0
    best_T_res = minimize(nll_with_temperature, initial_T, bounds=[(0.01, 100.0)], method='L-BFGS-B')
    best_T = best_T_res.x[0]
    
    print(f"Fitted Temperature: {best_T:.4f}")
    
    # Before Calibration Metrics
    nll_before = nll_with_temperature(1.0)
    
    pred_classes_before = np.argmax(probs_np, axis=1)
    acc_before = np.mean(pred_classes_before == y_true)
    
    # Brier score (multi-class)
    y_true_one_hot = np.eye(5)[y_true]
    brier_before = np.mean(np.sum((probs_np - y_true_one_hot)**2, axis=1))
    
    # ECE before (using max confidence)
    conf_before = np.max(probs_np, axis=1)
    ece_before = expected_calibration_error(pred_classes_before == y_true, conf_before)
    
    # After Calibration Metrics
    nll_after = nll_with_temperature(best_T)
    
    scaled_logits = logits / best_T
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    scaled_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    pred_classes_after = np.argmax(scaled_probs, axis=1)
    acc_after = np.mean(pred_classes_after == y_true)
    
    brier_after = np.mean(np.sum((scaled_probs - y_true_one_hot)**2, axis=1))
    
    conf_after = np.max(scaled_probs, axis=1)
    ece_after = expected_calibration_error(pred_classes_after == y_true, conf_after)
    
    print("\n--- Calibration Results ---")
    print(f"NLL   : Before = {nll_before:.4f}, After = {nll_after:.4f}")
    print(f"Brier : Before = {brier_before:.4f}, After = {brier_after:.4f}")
    print(f"ECE   : Before = {ece_before:.4f}, After = {ece_after:.4f}")
    print(f"Acc   : Before = {acc_before:.4f}, After = {acc_after:.4f}")
    
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # Include metadata
    manifest_hash = ""
    with open(manifest_path, "rb") as f:
        import hashlib
        manifest_hash = hashlib.sha256(f.read()).hexdigest()
        
    config = {
        "temperature": float(best_T),
        "fitted_status": True,
        "checkpoint_hash": "b2069ed07d2c31e9c5ed104e79391ab1e71ab85295cfccf00cf05cdaaf7284b8", 
        "calibration_manifest_hash": manifest_hash,
        "sample_count": len(df),
        "preprocessing_identity": "duplicate_div255_rgb_224x224"
    }
    
    with open(config_dir / "calibration.json", "w") as f:
        json.dump(config, f, indent=2)
        
    print("\nSaved calibration config to config/calibration.json")

if __name__ == "__main__":
    main()
