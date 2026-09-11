# iqa_module/test_iqa.py
import os
import sys
import glob

try:
    from .pipeline import process_retinal_image
except ImportError:
    from pipeline import process_retinal_image

if len(sys.argv) > 1:
    target = sys.argv[1]
    if os.path.isdir(target):
        image_paths = sorted(glob.glob(os.path.join(target, "IDRiD_*.jpg")) or glob.glob(os.path.join(target, "*.jpg")))
    else:
        image_paths = [target]
else:
    search_dirs = [
        r"C:\Users\HP\Desktop\testing",
        r"C:\Users\HP\Desktop",
    ]
    image_paths = []
    for d in search_dirs:
        found = sorted(glob.glob(os.path.join(d, "IDRiD_*.jpg")))
        if found:
            image_paths = found
            break

if not image_paths:
    print("No sample images found in Desktop or testing folder. Provide path via: python test_iqa.py <path_or_folder>")
    sys.exit(0)

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