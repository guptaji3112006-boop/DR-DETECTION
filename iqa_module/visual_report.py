# iqa_module/visual_report.py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
    from .pipeline import process_retinal_image
    from .quality_assessment import load_image, extract_fov_mask
except ImportError:
    from pipeline import process_retinal_image
    from quality_assessment import load_image, extract_fov_mask


def generate_visual_report(image_path, output_path=None):
    """
    Generate an infographic-style clinical report card for a retinal image.
    Shows Original, FOV Mask, Enhanced Image, and the Decision Scorecard.
    """
    if output_path is None:
        base = os.path.splitext(image_path)[0]
        output_path = f"{base}_quality_report.png"

    # 1. Run pipeline
    res = process_retinal_image(image_path)
    orig_img = load_image(image_path)
    mask = extract_fov_mask(orig_img)

    # Convert BGR to RGB for matplotlib display
    orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    enhanced_rgb = (
        cv2.cvtColor(res["processed_image"], cv2.COLOR_BGR2RGB)
        if res.get("processed_image") is not None
        else orig_rgb
    )

    # 2. Setup Figure (3 panels)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0F172A")  # Dark slate theme

    # Plot 1: Raw Original
    axes[0].imshow(orig_rgb)
    axes[0].set_title("1. Raw Camera Capture", color="white", fontsize=12, pad=10)
    axes[0].axis("off")

    # Plot 2: Retinal Mask
    fov_percent = res.get("metrics", {}).get("fov_coverage", 0.0) * 100
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title(f"2. Retinal Mask ({fov_percent:.1f}% FOV)", color="white", fontsize=12, pad=10)
    axes[1].axis("off")

    # Plot 3: Enhanced
    axes[2].imshow(enhanced_rgb)
    tag = "Enhanced (LAB-CLAHE)" if res.get("processed_image") is not None else "Unmodified (Rejected)"
    axes[2].set_title(f"3. {tag}", color="white", fontsize=12, pad=10)
    axes[2].axis("off")

    # 3. Overall Title & Clinical Guidance Banner
    status = res.get("status", "UNKNOWN")
    score = res.get("quality_score", 0.0)
    guidance = res.get("clinical_guidance", res.get("reason", "No guidance available."))
    status_color = "#10B981" if status == "PASS" else "#EF4444"

    fig.suptitle(
        f"Retinal Screening Quality: {status}  |  Score: {score:.2f} / 1.00\n"
        f"Guidance: {guidance}",
        color=status_color,
        fontsize=11,
        weight="bold",
        y=1.06,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"Report card generated -> {output_path}")
    return output_path


if __name__ == "__main__":
    import sys
    test_img = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\HP\Desktop\IDRiD_001.jpg"
    generate_visual_report(test_img)