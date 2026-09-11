import os
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LABELS_CSV = ROOT / "external_data" / "idrid" / "IDRiD_Testing_Labels.csv"
TESTING_IMAGES_DIR = (
    ROOT
    / "external_data"
    / "idrid"
    / "images"
    / "idrid_full_dataset"
    / "idrid"
    / "B. Disease Grading"
    / "1. Original Images"
    / "b. Testing Set"
)

MANIFEST_OUTPUT = ROOT / "validation_sim" / "results" / "test_manifest.csv"

def main():
    print("Preparing Cross-Dataset Validation Manifest...")

    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Labels CSV missing: {LABELS_CSV}")

    if not TESTING_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Testing images directory missing: {TESTING_IMAGES_DIR}")

    df = pd.read_csv(LABELS_CSV)
    
    # Strip whitespace from column names if any
    df.columns = df.columns.str.strip()
    
    # We expect 'Image name', 'Retinopathy grade'
    if "Image name" not in df.columns or "Retinopathy grade" not in df.columns:
        raise ValueError(f"Unexpected CSV columns: {df.columns}")

    manifest_records = []
    missing_images = []

    for index, row in df.iterrows():
        image_name = str(row["Image name"]).strip()
        # Ensure it has .jpg extension
        if not image_name.endswith(".jpg"):
            image_filename = f"{image_name}.jpg"
        else:
            image_filename = image_name
            
        image_path = TESTING_IMAGES_DIR / image_filename

        if image_path.exists():
            manifest_records.append({
                "image_name": image_filename,
                "image_path": str(image_path),
                "true_grade": int(row["Retinopathy grade"]),
                "dataset": "IDRiD",
                "split": "Testing"
            })
        else:
            missing_images.append(image_filename)

    if missing_images:
        print(f"WARNING: {len(missing_images)} images listed in CSV were not found in directory.")

    manifest_df = pd.DataFrame(manifest_records)
    
    # Check for duplicates
    if manifest_df.duplicated(subset=["image_name"]).any():
        print("WARNING: Duplicates found in manifest!")

    # Create results dir if not exists
    MANIFEST_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    
    manifest_df.to_csv(MANIFEST_OUTPUT, index=False)

    print(f"Manifest created with {len(manifest_df)} valid images.")
    print(f"Saved to: {MANIFEST_OUTPUT}")

if __name__ == "__main__":
    main()
