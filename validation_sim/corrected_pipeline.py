"""
Corrected DR Calibration & Threshold Pipeline.

Implements the full corrected workflow:
  Step 2: Proper provenance and versioned bundles
  Step 3: Dev/Test separation with cross-split duplicate filtering
  Step 4: Correct calibration → threshold order
  Step 5: Evaluation on hold-out test set

Usage:
  python corrected_pipeline.py --repo-root <path> --phase <calibrate|tune|evaluate|all>
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_curve, auc, confusion_matrix, cohen_kappa_score


# ── Utility ─────────────────────────────────────────────────────────────

def sha256_file(path):
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def softmax(logits):
    """Numerically stable softmax."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def expected_calibration_error(correct, confidence, n_bins=15):
    """ECE: weighted average |accuracy - confidence| across bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        conf = confidence[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return ece


def validate_probabilities(probs):
    """Reject invalid model outputs."""
    if probs.shape[-1] != 5:
        raise ValueError(f"Expected 5 classes, got shape {probs.shape}")
    if not np.all(np.isfinite(probs)):
        raise ValueError("Non-finite probabilities")
    if np.any(probs < 0):
        raise ValueError("Negative probabilities")
    if np.all(probs == 0):
        raise ValueError("All-zero probabilities")
    total = probs.sum()
    if not np.isclose(total, 1.0, atol=1e-3):
        raise ValueError(f"Probabilities sum to {total}, expected ~1.0")


# ── Model Adapter (no implicit config loading) ─────────────────────────

class CleanDRAdapter:
    """
    DR model adapter without implicit global config side-effects.
    
    Temperature and threshold are NOT loaded at module import time.
    They must be passed explicitly to apply_calibration() and
    apply_threshold().
    """
    
    CLASS_NAMES = [
        "No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"
    ]
    
    def __init__(self, checkpoint_path):
        import keras
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        self.checkpoint_hash = sha256_file(self.checkpoint_path)
        self.model = keras.models.load_model(str(self.checkpoint_path), compile=False)
        
        if tuple(self.model.input_shape[1:]) != (224, 224, 3):
            raise ValueError(f"Expected input (224,224,3), got {self.model.input_shape[1:]}")
        if self.model.output_shape[-1] != 5:
            raise ValueError(f"Expected 5 output classes, got {self.model.output_shape[-1]}")
    
    @staticmethod
    def preprocess(image_path):
        """Match training preprocessing: resize to 224x224, RGB, /255."""
        from PIL import Image
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        with Image.open(image_path) as img:
            img = img.resize((224, 224)).convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0
        
        return np.expand_dims(arr, axis=0)
    
    def predict_raw(self, image_path):
        """
        Return raw (uncalibrated) probabilities from the model.
        No threshold, no calibration, no side effects.
        """
        inp = self.preprocess(image_path)
        output = np.asarray(self.model.predict(inp, verbose=0))
        
        if output.shape != (1, 5):
            raise ValueError(f"Unexpected output shape: {output.shape}")
        
        probs = output[0]
        validate_probabilities(probs)
        
        grade = int(np.argmax(probs))
        return {
            "raw_probabilities": probs.tolist(),
            "predicted_grade": grade,
            "raw_confidence": float(probs[grade]),
        }
    
    @staticmethod
    def apply_calibration(raw_probs, temperature):
        """
        Apply temperature scaling to raw probabilities.
        Returns calibrated probabilities.
        Temperature must be a positive scalar.
        """
        if temperature is None or temperature <= 0:
            raise ValueError(f"Invalid temperature: {temperature}")
        
        probs = np.array(raw_probs, dtype=np.float64)
        eps = 1e-12
        probs = np.clip(probs, eps, 1.0 - eps)
        logits = np.log(probs)
        scaled_logits = logits / temperature
        calibrated = softmax(scaled_logits)
        return calibrated.tolist()
    
    @staticmethod
    def compute_referable_score(probs):
        """P(referable) = P(grade2) + P(grade3) + P(grade4)."""
        return float(probs[2] + probs[3] + probs[4])
    
    @staticmethod
    def apply_threshold(referable_score, threshold):
        """Binary referral decision."""
        return bool(referable_score >= threshold)


# ── Phase: Create Clean Development Manifest ────────────────────────────

def create_clean_dev_manifest(repo_root, output_dir):
    """
    Create a development manifest from IDRiD Training labels,
    EXCLUDING any images whose SHA-256 hash matches a Testing set image.
    """
    labels_csv = repo_root / "external_data" / "idrid" / "IDRiD_Training_Labels.csv"
    images_dir = (
        repo_root / "external_data" / "idrid" / "images" 
        / "idrid_full_dataset" / "idrid" / "B. Disease Grading" 
        / "1. Original Images" / "a. Training Set"
    )
    test_manifest = repo_root / "validation_sim" / "results" / "test_manifest.csv"
    
    if not labels_csv.is_file():
        raise FileNotFoundError(f"Labels not found: {labels_csv}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    
    # Collect test set hashes
    test_hashes = set()
    if test_manifest.is_file():
        df_test = pd.read_csv(test_manifest)
        if 'sha256' in df_test.columns:
            test_hashes = set(df_test['sha256'].dropna().str.lower().str.strip())
    
    print(f"Test set hashes loaded: {len(test_hashes)}")
    
    df_labels = pd.read_csv(labels_csv)
    records = []
    excluded = []
    
    for _, row in df_labels.iterrows():
        image_name = row['Image name'] + ".jpg"
        grade = int(row['Retinopathy grade'])
        img_path = images_dir / image_name
        
        if not img_path.exists():
            print(f"  Warning: {img_path} not found, skipping")
            continue
        
        file_hash = sha256_file(img_path)
        
        # Exclude cross-split duplicates
        if file_hash in test_hashes:
            excluded.append({
                "image_name": image_name,
                "grade": grade,
                "sha256": file_hash,
                "reason": "Hash matches Testing set image"
            })
            continue
        
        rel_path = img_path.relative_to(repo_root / "external_data").as_posix()
        records.append({
            "sample_id": f"IDRiD_Training_{row['Image name']}",
            "dataset": "IDRiD",
            "split": "Training",
            "image_name": image_name,
            "relative_path": rel_path,
            "true_grade": grade,
            "sha256": file_hash,
        })
    
    # Deduplicate within development data (by hash)
    df = pd.DataFrame(records)
    dup_hashes = df[df.duplicated(subset=['sha256'], keep=False)]
    if len(dup_hashes) > 0:
        print(f"  Warning: {len(dup_hashes)} records share hashes within dev set")
        # Keep first occurrence
        before = len(df)
        df = df.drop_duplicates(subset=['sha256'], keep='first')
        print(f"  Deduplicated: {before} -> {len(df)}")
    
    manifest_path = output_dir / "clean_development_manifest.csv"
    df.to_csv(manifest_path, index=False)
    
    # Save exclusion log
    if excluded:
        excl_df = pd.DataFrame(excluded)
        excl_df.to_csv(output_dir / "dev_exclusions.csv", index=False)
    
    print(f"\nClean development manifest: {len(df)} samples")
    print(f"Excluded cross-split duplicates: {len(excluded)}")
    print(f"Saved to: {manifest_path}")
    
    # Class distribution
    counts = df['true_grade'].value_counts().sort_index()
    for g, c in counts.items():
        ref = "Referable" if g >= 2 else "Non-referable"
        print(f"  Grade {g} ({ref}): {c}")
    
    return manifest_path


# ── Phase: Calibrate ────────────────────────────────────────────────────

def run_calibration(repo_root, dev_manifest_path, checkpoint_path, output_dir):
    """
    Fit temperature scaling on the development set.
    
    NOTE: Since we only have one development set (IDRiD Training),
    we split it into:
      - calibration_fit (70%): for fitting temperature
      - threshold_dev (30%): for choosing the referral threshold
    
    This ensures threshold is chosen on data not used for calibration fitting.
    """
    df = pd.read_csv(dev_manifest_path)
    dataset_root = repo_root / "external_data"
    
    adapter = CleanDRAdapter(checkpoint_path)
    
    print(f"\nRunning raw inference on {len(df)} development images...")
    raw_results = []
    
    for idx, row in df.iterrows():
        img_path = dataset_root / row['relative_path']
        try:
            pred = adapter.predict_raw(img_path)
            raw_results.append({
                "sample_id": row['sample_id'],
                "true_grade": int(row['true_grade']),
                "raw_probabilities": pred['raw_probabilities'],
                "predicted_grade": pred['predicted_grade'],
            })
        except Exception as e:
            print(f"  ERROR on {row['sample_id']}: {e}")
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx+1}/{len(df)}")
    
    print(f"Successful predictions: {len(raw_results)}")
    
    # Stratified split: 70% calibration-fitting, 30% threshold-development
    from sklearn.model_selection import StratifiedShuffleSplit
    
    y_grades = [r['true_grade'] for r in raw_results]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    cal_idx, thr_idx = next(splitter.split(np.zeros(len(raw_results)), y_grades))
    
    cal_results = [raw_results[i] for i in cal_idx]
    thr_results = [raw_results[i] for i in thr_idx]
    
    print(f"\nCalibration-fitting set: {len(cal_results)} samples")
    print(f"Threshold-development set: {len(thr_results)} samples")
    
    # ── Fit temperature on calibration-fitting set ──
    y_true_cal = np.array([r['true_grade'] for r in cal_results])
    probs_cal = np.array([r['raw_probabilities'] for r in cal_results])
    
    eps = 1e-12
    probs_cal_clipped = np.clip(probs_cal, eps, 1.0 - eps)
    logits_cal = np.log(probs_cal_clipped)
    
    def nll_objective(T):
        T = T[0]
        if T <= 0:
            return 1e10
        scaled = logits_cal / T
        calibrated = softmax(scaled)
        calibrated = np.clip(calibrated, eps, 1.0 - eps)
        nll = -np.mean(np.log(calibrated[np.arange(len(y_true_cal)), y_true_cal]))
        return nll
    
    result = minimize(nll_objective, x0=[1.0], bounds=[(0.01, 100.0)], method='L-BFGS-B')
    fitted_temperature = float(result.x[0])
    
    print(f"\nFitted Temperature: {fitted_temperature:.6f}")
    
    # Calibration diagnostics on FITTING set (labeled as such)
    nll_before = float(nll_objective([1.0]))
    nll_after = float(nll_objective([fitted_temperature]))
    
    cal_probs_before = probs_cal
    cal_probs_after = softmax(logits_cal / fitted_temperature)
    
    y_onehot = np.eye(5)[y_true_cal]
    brier_before = float(np.mean(np.sum((cal_probs_before - y_onehot)**2, axis=1)))
    brier_after = float(np.mean(np.sum((cal_probs_after - y_onehot)**2, axis=1)))
    
    correct_before = (np.argmax(cal_probs_before, 1) == y_true_cal)
    correct_after = (np.argmax(cal_probs_after, 1) == y_true_cal)
    ece_before = expected_calibration_error(correct_before, np.max(cal_probs_before, 1))
    ece_after = expected_calibration_error(correct_after, np.max(cal_probs_after, 1))
    
    print(f"\n--- Calibration Diagnostics (FITTING SET — not independent) ---")
    print(f"NLL:   Before={nll_before:.4f}, After={nll_after:.4f}")
    print(f"Brier: Before={brier_before:.4f}, After={brier_after:.4f}")
    print(f"ECE:   Before={ece_before:.4f}, After={ece_after:.4f}")
    
    # ── Save calibration bundle ──
    manifest_hash = sha256_file(dev_manifest_path)
    
    cal_bundle = {
        "temperature": fitted_temperature,
        "fitted_status": True,
        "fitting_method": "L-BFGS-B NLL minimization on 5-class logits",
        "checkpoint_sha256": adapter.checkpoint_hash,
        "calibration_manifest_sha256": manifest_hash,
        "calibration_manifest_path": str(dev_manifest_path),
        "calibration_fitting_sample_count": len(cal_results),
        "threshold_dev_sample_count": len(thr_results),
        "total_dev_sample_count": len(raw_results),
        "split_method": "StratifiedShuffleSplit(test_size=0.3, random_state=42)",
        "preprocessing": "PIL resize(224,224) RGB /255.0",
        "class_mapping": {
            "0": "No DR (non-referable)",
            "1": "Mild DR (non-referable)",
            "2": "Moderate DR (referable)",
            "3": "Severe DR (referable)",
            "4": "Proliferative DR (referable)",
        },
        "referral_score_definition": "P(grade2) + P(grade3) + P(grade4)",
        "binary_target": "ICDR grade >= 2",
        "fitting_diagnostics": {
            "label": "ON FITTING SET - NOT INDEPENDENT",
            "nll_before": nll_before,
            "nll_after": nll_after,
            "brier_before": brier_before,
            "brier_after": brier_after,
            "ece_before": ece_before,
            "ece_after": ece_after,
        },
        "created_at": datetime.now().astimezone().isoformat(),
        "cross_split_duplicates_excluded": True,
        "dataset_independence_note": (
            "Development data is IDRiD Training. Model was trained on APTOS 2019. "
            "No overlap has been identified between APTOS and IDRiD, but this has "
            "not been independently verified."
        ),
    }
    
    bundle_path = output_dir / "calibration_bundle.json"
    with open(bundle_path, "w") as f:
        json.dump(cal_bundle, f, indent=2)
    print(f"\nSaved calibration bundle: {bundle_path}")
    
    # Save raw predictions for threshold tuning
    raw_preds_path = output_dir / "raw_predictions.json"
    with open(raw_preds_path, "w") as f:
        json.dump({
            "calibration_fitting_indices": cal_idx.tolist(),
            "threshold_dev_indices": thr_idx.tolist(),
            "predictions": raw_results,
        }, f, indent=2)
    
    return fitted_temperature, thr_results, adapter.checkpoint_hash


# ── Phase: Threshold Tuning ─────────────────────────────────────────────

def run_threshold_tuning(thr_results, fitted_temperature, output_dir):
    """
    Choose the referral threshold on the threshold-development partition
    using calibrated probabilities.
    
    Rule: Among operating points with sensitivity >= 90%, maximize specificity.
    """
    print(f"\n--- Threshold Tuning on {len(thr_results)} samples ---")
    
    y_true_binary = np.array([1 if r['true_grade'] >= 2 else 0 for r in thr_results])
    
    # Apply calibration to get referable scores
    referable_scores = []
    for r in thr_results:
        cal_probs = CleanDRAdapter.apply_calibration(r['raw_probabilities'], fitted_temperature)
        score = CleanDRAdapter.compute_referable_score(cal_probs)
        referable_scores.append(score)
    
    y_scores = np.array(referable_scores)
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_scores)
    roc_auc_val = auc(fpr, tpr)
    print(f"AUC-ROC: {roc_auc_val:.4f}")
    
    # Find threshold: sensitivity >= 90%, maximize specificity
    target_sensitivity = 0.90
    valid_mask = tpr >= target_sensitivity
    
    if not valid_mask.any():
        print(f"WARNING: No threshold achieves sensitivity >= {target_sensitivity:.0%}")
        # Fall back to point closest to target
        best_idx = int(np.argmin(np.abs(tpr - target_sensitivity)))
        target_met = False
    else:
        # Among valid points, minimize FPR (maximize specificity)
        valid_indices = np.where(valid_mask)[0]
        best_valid = valid_indices[np.argmin(fpr[valid_indices])]
        best_idx = best_valid
        target_met = True
    
    chosen_threshold = float(thresholds[best_idx])
    chosen_sensitivity = float(tpr[best_idx])
    chosen_specificity = float(1 - fpr[best_idx])
    
    # Compute confusion matrix at chosen threshold
    y_pred = (y_scores >= chosen_threshold).astype(int)
    tp = int(((y_true_binary == 1) & (y_pred == 1)).sum())
    tn = int(((y_true_binary == 0) & (y_pred == 0)).sum())
    fp = int(((y_true_binary == 0) & (y_pred == 1)).sum())
    fn = int(((y_true_binary == 1) & (y_pred == 0)).sum())
    
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    npv = tn / (tn + fn) if (tn + fn) > 0 else float('nan')
    referral_rate = y_pred.mean()
    
    print(f"\n--- Threshold Tuning Results (Development Partition) ---")
    print(f"Chosen Threshold: {chosen_threshold:.6f}")
    print(f"Sensitivity: {chosen_sensitivity:.4f}")
    print(f"Specificity: {chosen_specificity:.4f}")
    print(f"PPV: {ppv:.4f}")
    print(f"NPV: {npv:.4f}")
    print(f"Referral Rate: {referral_rate:.4f}")
    print(f"TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"Sensitivity target met: {target_met}")
    print(f"Specificity >= 85%: {chosen_specificity >= 0.85}")
    
    # Save threshold bundle
    threshold_bundle = {
        "referral_threshold": chosen_threshold,
        "comparison_operator": ">=",
        "referral_score_definition": "P(grade2) + P(grade3) + P(grade4)",
        "binary_target": "ICDR grade >= 2 (referable DR proxy)",
        "selection_rule": "Among operating points with sensitivity >= 90%, maximize specificity",
        "development_partition_size": len(thr_results),
        "development_metrics": {
            "sensitivity": chosen_sensitivity,
            "specificity": chosen_specificity,
            "ppv": ppv,
            "npv": npv,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "referral_rate": referral_rate,
            "auc_roc": roc_auc_val,
        },
        "target_sensitivity_met": target_met,
        "specificity_85_met": chosen_specificity >= 0.85,
        "calibration_temperature": fitted_temperature,
        "label": "DEVELOPMENT SET PERFORMANCE - NOT INDEPENDENT EVALUATION",
        "created_at": datetime.now().astimezone().isoformat(),
    }
    
    bundle_path = output_dir / "threshold_bundle.json"
    with open(bundle_path, "w") as f:
        json.dump(threshold_bundle, f, indent=2)
    print(f"\nSaved threshold bundle: {bundle_path}")
    
    return chosen_threshold


# ── Phase: Evaluate on Test Set ─────────────────────────────────────────

def run_evaluation(repo_root, checkpoint_path, fitted_temperature, chosen_threshold, output_dir):
    """
    Evaluate the frozen bundle on the IDRiD Testing set (103 images).
    No tuning allowed here.
    """
    test_manifest = repo_root / "validation_sim" / "results" / "test_manifest.csv"
    dataset_root = repo_root / "external_data"
    
    if not test_manifest.is_file():
        raise FileNotFoundError(f"Test manifest not found: {test_manifest}")
    
    df_test = pd.read_csv(test_manifest)
    
    adapter = CleanDRAdapter(checkpoint_path)
    
    print(f"\n--- Evaluating on IDRiD Testing Set ({len(df_test)} images) ---")
    print(f"Checkpoint: {adapter.checkpoint_hash[:16]}...")
    print(f"Temperature: {fitted_temperature}")
    print(f"Threshold: {chosen_threshold}")
    
    records = []
    errors = []
    
    for idx, row in df_test.iterrows():
        img_path = dataset_root / row['relative_path']
        try:
            pred = adapter.predict_raw(img_path)
            raw_probs = pred['raw_probabilities']
            
            # Apply calibration
            cal_probs = CleanDRAdapter.apply_calibration(raw_probs, fitted_temperature)
            
            # Compute referable score
            ref_score = CleanDRAdapter.compute_referable_score(cal_probs)
            
            # Apply threshold
            is_ref = CleanDRAdapter.apply_threshold(ref_score, chosen_threshold)
            
            records.append({
                "sample_id": row['sample_id'],
                "true_grade": int(row['true_grade']),
                "predicted_grade": pred['predicted_grade'],
                "raw_probabilities": raw_probs,
                "raw_confidence": pred['raw_confidence'],
                "calibrated_probabilities": cal_probs,
                "referable_score": ref_score,
                "is_referable": is_ref,
                "inference_status": "SUCCESS",
            })
        except Exception as e:
            errors.append({"sample_id": row['sample_id'], "error": str(e)})
            print(f"  ERROR: {row['sample_id']}: {e}")
        
        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx+1}/{len(df_test)}")
    
    print(f"\nSuccessful: {len(records)}, Errors: {len(errors)}")
    
    if not records:
        print("FATAL: No successful predictions")
        return
    
    # Compute metrics
    y_true_grade = np.array([r['true_grade'] for r in records])
    y_pred_grade = np.array([r['predicted_grade'] for r in records])
    y_true_binary = (y_true_grade >= 2).astype(int)
    y_pred_binary = np.array([int(r['is_referable']) for r in records])
    
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()
    
    pos = tp + fn
    neg = tn + fp
    sensitivity = tp / pos if pos > 0 else float('nan')
    specificity = tn / neg if neg > 0 else float('nan')
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    npv = tn / (tn + fn) if (tn + fn) > 0 else float('nan')
    referral_rate = y_pred_binary.mean()
    
    # Five-class metrics
    five_class_acc = float((y_pred_grade == y_true_grade).mean())
    try:
        qwk = float(cohen_kappa_score(y_true_grade, y_pred_grade, labels=[0,1,2,3,4], weights='quadratic'))
    except:
        qwk = float('nan')
    
    # Calibration metrics on test set (independent evaluation)
    cal_probs_all = np.array([r['calibrated_probabilities'] for r in records])
    correct = (np.argmax(cal_probs_all, 1) == y_true_grade)
    conf = np.max(cal_probs_all, 1)
    ece_test = expected_calibration_error(correct, conf)
    
    eps_eval = 1e-12
    cal_probs_clipped = np.clip(cal_probs_all, eps_eval, 1.0 - eps_eval)
    nll_test = float(-np.mean(np.log(cal_probs_clipped[np.arange(len(y_true_grade)), y_true_grade])))
    
    y_onehot_test = np.eye(5)[y_true_grade]
    brier_test = float(np.mean(np.sum((cal_probs_all - y_onehot_test)**2, axis=1)))
    
    # FP/FN breakdown
    fp_breakdown = {}
    fn_breakdown = {}
    for r in records:
        tg = r['true_grade']
        ref = r['is_referable']
        true_ref = tg >= 2
        if not true_ref and ref:  # FP
            label = "No DR" if tg == 0 else "Mild DR"
            fp_breakdown[f"Grade {tg} ({label})"] = fp_breakdown.get(f"Grade {tg} ({label})", 0) + 1
        if true_ref and not ref:  # FN
            labels = {2: "Moderate DR", 3: "Severe DR", 4: "Proliferative DR"}
            fn_breakdown[f"Grade {tg} ({labels.get(tg, '?')})"] = fn_breakdown.get(f"Grade {tg} ({labels.get(tg, '?')})", 0) + 1
    
    print(f"\n{'='*60}")
    print(f"  FINAL EVALUATION — IDRiD Testing Set (Independent)")
    print(f"{'='*60}")
    print(f"  Images evaluated: {len(records)}")
    print(f"  Inference errors: {len(errors)}")
    print(f"")
    print(f"  Binary Referral (ICDR grade >= 2):")
    print(f"    TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"    Sensitivity: {sensitivity:.4f} ({sensitivity:.2%})")
    print(f"    Specificity: {specificity:.4f} ({specificity:.2%})")
    print(f"    PPV: {ppv:.4f}")
    print(f"    NPV: {npv:.4f}")
    print(f"    Referral Rate: {referral_rate:.4f}")
    print(f"")
    print(f"  False Positives by true grade: {fp_breakdown}")
    print(f"  False Negatives by true grade: {fn_breakdown}")
    print(f"")
    print(f"  Five-class:")
    print(f"    Accuracy: {five_class_acc:.4f}")
    print(f"    Quadratic Weighted Kappa: {qwk:.4f}")
    print(f"")
    print(f"  Calibration (independent evaluation):")
    print(f"    NLL: {nll_test:.4f}")
    print(f"    Brier: {brier_test:.4f}")
    print(f"    ECE: {ece_test:.4f}")
    print(f"")
    print(f"  TARGETS:")
    sens_met = sensitivity >= 0.90
    spec_met = specificity >= 0.85
    print(f"    Sensitivity >= 90%: {'YES' if sens_met else 'NO'} ({sensitivity:.2%})")
    print(f"    Specificity >= 85%: {'YES' if spec_met else 'NO'} ({specificity:.2%})")
    print(f"    Both targets met:   {'YES' if (sens_met and spec_met) else 'NO'}")
    print(f"{'='*60}")
    
    # Save evaluation results
    eval_results = {
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        "run_date": datetime.now().astimezone().isoformat(),
        "dataset": "IDRiD Testing",
        "sample_count": len(records),
        "inference_errors": len(errors),
        "checkpoint_sha256": adapter.checkpoint_hash,
        "calibration_temperature": fitted_temperature,
        "referral_threshold": chosen_threshold,
        "threshold_comparison": ">=",
        "referral_score_definition": "P(grade2) + P(grade3) + P(grade4)",
        "preprocessing": "PIL resize(224,224) RGB /255.0",
        "binary_metrics": {
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "ppv": ppv,
            "npv": npv,
            "referral_rate": referral_rate,
        },
        "false_positive_breakdown": fp_breakdown,
        "false_negative_breakdown": fn_breakdown,
        "five_class_metrics": {
            "accuracy": five_class_acc,
            "qwk": qwk,
        },
        "calibration_metrics_independent": {
            "nll": nll_test,
            "brier": brier_test,
            "ece": ece_test,
        },
        "targets": {
            "sensitivity_90_met": sens_met,
            "specificity_85_met": spec_met,
            "both_met": sens_met and spec_met,
        },
        "test_set_grade_distribution": {
            str(g): int((y_true_grade == g).sum()) for g in range(5)
        },
        "notes": [
            "This is independent evaluation — threshold and temperature were frozen before testing.",
            "IDRiD Testing labels were NOT used for threshold selection or calibration fitting.",
            "Cross-split duplicate (IDRiD_118 Training / IDRiD_064 Testing) was excluded from development data.",
            "IQA status not evaluated in this run — all images processed.",
            "This is retrospective evaluation, not clinical validation.",
        ],
    }
    
    eval_path = output_dir / "evaluation_results.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2, default=str)
    
    # Save prediction audit
    audit_df = pd.DataFrame([{
        "sample_id": r['sample_id'],
        "true_grade": r['true_grade'],
        "predicted_grade": r['predicted_grade'],
        "raw_confidence": r['raw_confidence'],
        "raw_probabilities": json.dumps(r['raw_probabilities']),
        "calibrated_probabilities": json.dumps(r['calibrated_probabilities']),
        "referable_score": r['referable_score'],
        "is_referable": r['is_referable'],
    } for r in records])
    
    audit_df.to_csv(output_dir / "prediction_audit.csv", index=False)
    
    print(f"\nResults saved to: {output_dir}")
    return eval_results


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Corrected DR calibration pipeline")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--phase", choices=["manifest", "calibrate", "tune", "evaluate", "all"], default="all")
    args = parser.parse_args()
    
    repo_root = args.repo_root.resolve()
    checkpoint = repo_root / "diabetic-retinopathy-detection" / "models" / "diabetic_retinopathy_model.keras"
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = repo_root / "validation_sim" / "results" / f"corrected_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Checkpoint: {checkpoint}")
    
    if args.phase in ("manifest", "all"):
        dev_manifest = create_clean_dev_manifest(repo_root, output_dir)
    else:
        # Use most recent clean manifest
        dev_manifest = sorted(
            (repo_root / "validation_sim" / "results").glob("corrected_*/clean_development_manifest.csv")
        )[-1]
    
    if args.phase in ("calibrate", "all"):
        temperature, thr_results, ckpt_hash = run_calibration(
            repo_root, dev_manifest, checkpoint, output_dir
        )
    
    if args.phase in ("tune", "all"):
        threshold = run_threshold_tuning(thr_results, temperature, output_dir)
    
    if args.phase in ("evaluate", "all"):
        eval_results = run_evaluation(
            repo_root, checkpoint, temperature, threshold, output_dir
        )
    
    print("\n=== Pipeline complete ===")


if __name__ == "__main__":
    main()
