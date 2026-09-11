# iqa_module/batch_evaluate.py
import os
import glob
import pandas as pd

try:
    from .pipeline import process_retinal_image
except ImportError:
    from pipeline import process_retinal_image


def run_batch_triage(image_folder):
    extensions = ("*.jpg", "*.jpeg", "*.png")
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))
    image_files = sorted(image_files)

    if not image_files:
        print(f"No images found in {image_folder}")
        return

    print(f"Running automated triage on {len(image_files)} retinal images...\n")
    records = []

    for img_path in image_files:
        res = process_retinal_image(img_path)
        m = res["metrics"]
        records.append({
            "Filename": os.path.basename(img_path),
            "Status": res["status"],
            "Quality_Score": res["quality_score"],
            "Sharpness": round(m["sharpness"], 1),
            "Brightness": round(m["brightness"], 1),
            "Contrast": round(m["contrast"], 1),
            "FOV_Coverage": f"{m['fov_coverage']*100:.1f}%",
            "Uniformity": round(m["uniformity"], 2),
            "Glare_Ratio": f"{m['glare_ratio']*100:.2f}%",
            "Reason": res["reason"],
            "Actionable_Guidance": res["clinical_guidance"],
        })

    df = pd.DataFrame(records)
    csv_path = os.path.join(image_folder, "iqa_triage_report.csv")
    df.to_csv(csv_path, index=False)

    total = len(df)
    passed = (df["Status"] == "PASS").sum()
    rejected = (df["Status"] == "REJECT").sum()

    print("=" * 65)
    print("             BATCH TRIAGE SUMMARY REPORT")
    print("=" * 65)
    print(f"Total Evaluated       : {total}")
    print(f"Accepted for Grading  : {passed} ({passed/total*100:.1f}%)")
    print(f"Rejected for Retake   : {rejected} ({rejected/total*100:.1f}%)")
    print(f"Full Report Exported  : {csv_path}")
    print("=" * 65)


if __name__ == "__main__":
    import sys
    # Defaults to Desktop where sample images are located
    folder = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\HP\Desktop"
    run_batch_triage(folder)