import argparse
import pandas as pd
import hashlib
from pathlib import Path
import sys

def compute_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Generate a reproducible dataset manifest.")
    parser.add_argument("--labels", required=True, type=Path, help="Path to the official labels CSV.")
    parser.add_argument("--image-dir", required=True, type=Path, help="Directory containing the original images.")
    parser.add_argument("--dataset-name", required=True, type=str, help="Name of the dataset (e.g., IDRiD).")
    parser.add_argument("--split", required=True, type=str, help="Dataset split (e.g., Testing, Training).")
    parser.add_argument("--dataset-root", required=True, type=Path, help="Root directory for relative paths.")
    parser.add_argument("--output", required=True, type=Path, help="Output path for the manifest CSV.")
    
    args = parser.parse_args()

    if not args.labels.is_file():
        sys.exit(f"ERROR: Labels CSV not found at {args.labels}")
    if not args.image_dir.is_dir():
        sys.exit(f"ERROR: Image directory not found at {args.image_dir}")

    try:
        df = pd.read_csv(args.labels)
    except Exception as e:
        sys.exit(f"ERROR: Could not read labels CSV: {e}")

    df.columns = df.columns.str.strip()
    
    if "Image name" not in df.columns or "Retinopathy grade" not in df.columns:
        sys.exit(f"ERROR: Missing required columns 'Image name' or 'Retinopathy grade'. Found: {list(df.columns)}")

    # Check for duplicate image names
    duplicates = df[df.duplicated(subset=['Image name'], keep=False)]
    if not duplicates.empty:
        sys.exit(f"ERROR: Duplicate records found in labels CSV:\n{duplicates}")

    manifest_records = []
    failures = []

    for idx, row in df.iterrows():
        image_name = str(row["Image name"]).strip()
        raw_grade = row["Retinopathy grade"]

        # Validate missing grade
        if pd.isna(raw_grade):
            failures.append((image_name, "Missing Retinopathy grade"))
            continue
            
        # Validate grade is integer 0-4 BEFORE casting
        if isinstance(raw_grade, float) and not raw_grade.is_integer():
            failures.append((image_name, f"Fractional grade found: {raw_grade}"))
            continue
            
        try:
            grade_str = str(raw_grade)
            if '.' in grade_str and not grade_str.endswith('.0'):
                failures.append((image_name, f"Fractional grade string found: {grade_str}"))
                continue
            grade = int(float(raw_grade))
        except ValueError:
            failures.append((image_name, f"Invalid grade format: {raw_grade}"))
            continue

        if grade not in [0, 1, 2, 3, 4]:
            failures.append((image_name, f"Grade out of range 0-4: {grade}"))
            continue

        # Find image file
        possible_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        found_files = []
        for ext in possible_extensions:
            if image_name.lower().endswith(ext):
                candidate = args.image_dir / image_name
            else:
                candidate = args.image_dir / f"{image_name}{ext}"
                
            if candidate.is_file():
                found_files.append(candidate)

        if len(found_files) == 0:
            failures.append((image_name, "Image file not found"))
            continue
        if len(found_files) > 1:
            failures.append((image_name, f"Ambiguous image match: {found_files}"))
            continue

        image_path = found_files[0]
        
        # Exclude generated reports or overlays based on filename heuristics
        if "overlay" in image_path.name.lower() or "report" in image_path.name.lower():
            failures.append((image_name, "Image appears to be a generated report or overlay"))
            continue

        # Create portable path
        try:
            portable_path = image_path.relative_to(args.dataset_root)
        except ValueError:
            failures.append((image_name, f"Image path {image_path} not relative to dataset root {args.dataset_root}"))
            continue

        sha256 = compute_sha256(image_path)
        
        # Unique identifier
        sample_id = f"{args.dataset_name}_{args.split}_{image_path.name}"

        record = {
            "sample_id": sample_id,
            "dataset": args.dataset_name,
            "split": args.split,
            "image_name": image_path.name,
            "relative_path": str(portable_path).replace("\\", "/"),
            "true_grade": grade,
            "sha256": sha256
        }
        
        # Validate Patient ID if supplied
        if "Patient ID" in df.columns:
            patient_id = row["Patient ID"]
            if pd.isna(patient_id):
                failures.append((image_name, "Missing Patient ID"))
                continue
            record["patient_id"] = str(patient_id).strip()
            
        manifest_records.append(record)

    print(f"--- Audit Report ---")
    print(f"Input rows: {len(df)}")
    print(f"Matched images: {len(manifest_records)}")
    print(f"Failures: {len(failures)}")
    
    if failures:
        print("\nFailure Details:")
        for img, reason in failures:
            print(f"  - {img}: {reason}")
        sys.exit("\nERROR: Manifest generation failed due to validation errors. Records will not be silently dropped.")

    if not manifest_records:
        sys.exit("ERROR: No valid records found.")

    manifest_df = pd.DataFrame(manifest_records)
    
    # Check for sample_id collisions
    if manifest_df.duplicated(subset=['sample_id']).any():
        sys.exit("ERROR: Generated sample_ids are not unique.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(args.output, index=False)
    print(f"\nSuccessfully saved manifest with {len(manifest_df)} records to {args.output}")

if __name__ == "__main__":
    main()
