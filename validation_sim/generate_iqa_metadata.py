import pandas as pd
import json
import sys
import os
from pathlib import Path

# Important: ensure DR-DETECTION is in sys.path
sys.path.insert(0, os.path.abspath('.'))

from validation_sim.evaluate import resolve_image_path
from iqa_module.pipeline import process_retinal_image

def generate(manifest_path, output_path):
    df = pd.read_csv(manifest_path)
    dataset_root = Path('external_data').resolve()
    
    results = []
    
    for _, row in df.iterrows():
        try:
            image_path = resolve_image_path(dataset_root, row["relative_path"])
            iqa_result = process_retinal_image(str(image_path))
            
            # metrics dictionary from assess_quality contains uniformity
            uniformity = iqa_result["metrics"].get("uniformity", "N/A")
            
            results.append({
                "sample_id": row["sample_id"],
                "quality_score": iqa_result["quality_score"],
                "uniformity": uniformity,
                "status": iqa_result["status"],
                "reason": iqa_result["reason"]
            })
            print(f"Processed {row['sample_id']}")
        except Exception as e:
            print(f"Error processing {row['sample_id']}: {e}")
            
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Saved {len(results)} records to {output_path}")

if __name__ == "__main__":
    generate("validation_sim/results/test_manifest.csv", sys.argv[1])
