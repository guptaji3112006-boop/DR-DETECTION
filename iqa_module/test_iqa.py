# iqa_module/test_iqa.py
import os
import glob
from pipeline import process_retinal_image

image_paths = sorted(glob.glob(r"C:\Users\HP\Desktop\IDRiD_*.jpg"))

print("=" * 115)
print(f"{'IMAGE':<14} | {'STATUS':<6} | {'SCORE':<5} | {'SHARP':<6} | {'BRIGHT':<6} | {'UNIFORM':<7} | {'GLARE':<6} | ACTIONABLE CLINICAL GUIDANCE")
print("=" * 115)

for path in image_paths:
    res = process_retinal_image(path)
    m = res["metrics"]
    img_name = os.path.basename(path)

    print(
        f"{img_name:<14} | "
        f"{res['status']:<6} | "
        f"{res['quality_score']:<5.2f} | "
        f"{m['sharpness']:<6.1f} | "
        f"{m['brightness']:<6.1f} | "
        f"{m['uniformity']:<7.2f} | "
        f"{m['glare_ratio']*100:<5.2f}% | "
        f"{res['clinical_guidance']}"
    )
print("=" * 115)