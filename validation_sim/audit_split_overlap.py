import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def get_file_hash(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def find_image_file(image_dir: Path, image_name: str) -> Path:
    base_name = Path(image_name).stem
    
    matches = list(image_dir.glob(f"{base_name}.*"))
    if not matches:
        raise FileNotFoundError(f"Image missing for {image_name} in {image_dir}")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous image match for {image_name} in {image_dir}: {matches}")
    
    return matches[0]


def process_split(labels_csv: Path, image_dir: Path, split_name: str) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    
    # Ignore wholly empty unnamed columns
    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
    empty_unnamed = [c for c in unnamed_cols if df[c].isna().all()]
    if empty_unnamed:
        df = df.drop(columns=empty_unnamed)

    # Determine ID and Grade columns
    id_col = None
    for col in ["Image name", "image_name", "sample_id", "Image_name", "Image"]:
        if col in df.columns:
            id_col = col
            break
            
    grade_col = None
    for col in ["Retinopathy grade", "true_grade", "Retinopathy_grade", "grade", "Severity", "severity_grade"]:
        if col in df.columns:
            grade_col = col
            break
            
    if not id_col or not grade_col:
        raise ValueError(f"Could not find required ID or Grade columns in {labels_csv}. Columns found: {list(df.columns)}")

    # Check for missing values
    if df[id_col].isna().any():
        raise ValueError(f"Missing values found in ID column '{id_col}' of {labels_csv}")
    if df[grade_col].isna().any():
        raise ValueError(f"Missing values found in Grade column '{grade_col}' of {labels_csv}")
        
    # Check for duplicate IDs
    if df[id_col].duplicated().any():
        duplicates = df[df[id_col].duplicated()][id_col].tolist()
        raise ValueError(f"Duplicate label IDs found in {labels_csv}: {duplicates}")

    # Validate grades
    try:
        df["_parsed_grade"] = pd.to_numeric(df[grade_col], errors="raise")
    except Exception as e:
        raise ValueError(f"Grade column '{grade_col}' contains non-numeric values in {labels_csv}") from e
        
    if not df["_parsed_grade"].isin([0, 1, 2, 3, 4]).all():
        raise ValueError(f"Grades must be integers 0-4. Invalid values found in {labels_csv}")

    records = []
    for _, row in df.iterrows():
        img_id = str(row[id_col]).strip()
        img_path = find_image_file(image_dir, img_id)
        
        file_hash = get_file_hash(img_path)
        
        records.append({
            "split": split_name,
            "image_name": img_path.name,
            "relative_path": str(img_path.relative_to(image_dir)),
            "true_grade": int(row["_parsed_grade"]),
            "sha256": file_hash,
            "original_id": img_id
        })
        
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Exact-file overlap audit between train and test splits.")
    parser.add_argument("--train-labels", type=Path, required=True, help="Path to training labels CSV")
    parser.add_argument("--train-image-dir", type=Path, required=True, help="Directory containing training images")
    parser.add_argument("--test-labels", type=Path, required=True, help="Path to testing labels CSV")
    parser.add_argument("--test-image-dir", type=Path, required=True, help="Directory containing testing images")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save audit outputs")
    
    args = parser.parse_args()
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / f"audit_overlap_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "run_date": datetime.now().isoformat(),
        "status": "INCOMPLETE",
        "train_count": 0,
        "test_count": 0,
        "exact_duplicates_found": 0,
        "errors": []
    }
    
    try:
        print("Processing training split...")
        train_df = process_split(args.train_labels, args.train_image_dir, "Train")
        summary["train_count"] = len(train_df)
        train_df.drop(columns=["original_id"]).to_csv(out_dir / "train_manifest.csv", index=False)
        
        print("Processing testing split...")
        test_df = process_split(args.test_labels, args.test_image_dir, "Test")
        summary["test_count"] = len(test_df)
        test_df.drop(columns=["original_id"]).to_csv(out_dir / "test_manifest.csv", index=False)
        
        # Check overlap by SHA-256
        print("Checking for exact-file overlaps...")
        merged = pd.merge(
            train_df, 
            test_df, 
            on="sha256", 
            suffixes=("_train", "_test")
        )
        
        duplicates = merged[["sha256", "image_name_train", "relative_path_train", "true_grade_train", 
                             "image_name_test", "relative_path_test", "true_grade_test"]]
        
        duplicates.to_csv(out_dir / "exact_duplicate_pairs.csv", index=False)
        
        num_dups = len(duplicates)
        summary["exact_duplicates_found"] = num_dups
        summary["status"] = "SUCCESS"
        summary["note"] = "No matching hashes does not exclude re-encoded copies, same-patient overlap, or tuning exposure."
        
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
            
        print(f"Audit complete. {num_dups} exact overlaps found.")
        print(f"Results saved to: {out_dir}")
        
        if num_dups > 0:
            sys.exit(1)
            
    except Exception as e:
        summary["status"] = "ERROR"
        summary["errors"].append(str(e))
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
