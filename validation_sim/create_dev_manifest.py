import pandas as pd
import hashlib
from pathlib import Path

# Paths
labels_csv = Path(r"external_data\idrid\IDRiD_Training_Labels.csv")
images_dir = Path(r"external_data\idrid\images\idrid_full_dataset\idrid\B. Disease Grading\1. Original Images\a. Training Set")
output_csv = Path(r"validation_sim\results\development_manifest.csv")

def get_sha256(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

df = pd.read_csv(labels_csv)

records = []
for idx, row in df.iterrows():
    image_name = row['Image name'] + ".jpg"
    grade = int(row['Retinopathy grade'])
    
    img_path = images_dir / image_name
    if not img_path.exists():
        print(f"Warning: {img_path} not found.")
        continue
        
    rel_path = img_path.relative_to(Path("external_data")).as_posix()
    
    records.append({
        "sample_id": f"IDRiD_Training_{row['Image name']}",
        "dataset": "IDRiD",
        "split": "Training",
        "image_name": image_name,
        "relative_path": rel_path,
        "true_grade": grade,
        "sha256": get_sha256(img_path)
    })

manifest_df = pd.DataFrame(records)
output_csv.parent.mkdir(parents=True, exist_ok=True)
manifest_df.to_csv(output_csv, index=False)
print(f"Saved {len(manifest_df)} records to {output_csv}")
