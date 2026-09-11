"""
Classical image-processing lesion detection + overlay.
Detects dark spots (hemorrhages/microaneurysms) and bright spots (exudates)
without needing a trained neural network.
"""

import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
FILE_NAMES  = ['lesion_nodr', 'lesion_mild', 'lesion_moderate',
               'lesion_severe', 'lesion_proliferative']
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'images')

print("Loading sample data...")
images = np.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'sample_images.npy'))
labels = np.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'sample_labels.npy'))
print(f"Loaded {len(images)} samples.")

def detect_lesions(img):
    """img: RGB float array 0-1 or 0-255. Returns overlay image with lesions circled."""
    img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Dark lesions (hemorrhages / microaneurysms)
    _, dark_thresh = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark_thresh = cv2.erode(dark_thresh, np.ones((3,3), np.uint8), iterations=1)

    # Bright lesions (exudates)
    _, bright_thresh = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright_thresh = cv2.erode(bright_thresh, np.ones((3,3), np.uint8), iterations=1)

    overlay = img_uint8.copy()
    lesion_count = 0

    for mask, color in [(dark_thresh, (255, 0, 0)), (bright_thresh, (255, 255, 0))]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if 5 < area < 300:  # filter out noise and huge blobs (like the whole eye outline)
                (x, y), r = cv2.minEnclosingCircle(c)
                cv2.circle(overlay, (int(x), int(y)), int(r)+2, color, 1)
                lesion_count += 1

    return overlay, lesion_count

os.makedirs(OUT_DIR, exist_ok=True)

for cls_idx in range(5):
    idxs = np.where(labels.astype(int) == cls_idx)[0]
    if len(idxs) == 0:
        print(f"WARNING: no samples for class {cls_idx} ({CLASS_NAMES[cls_idx]}), skipping.")
        continue

    img = images[idxs[0]]
    overlay, lesion_count = detect_lesions(img)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor('#080b12')

    axes[0].imshow(img)
    axes[0].set_title('Original Retinal Scan', color='white', fontsize=12, pad=10)
    axes[0].axis('off')

    axes[1].imshow(overlay)
    axes[1].set_title(
        f'Lesion Overlay\n{CLASS_NAMES[cls_idx]} · {lesion_count} regions flagged',
        color='white', fontsize=12, pad=10
    )
    axes[1].axis('off')

    plt.tight_layout(pad=1.5)
    out_path = os.path.join(OUT_DIR, f'{FILE_NAMES[cls_idx]}.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='#080b12')
    plt.close()

    print(f"[{cls_idx}] {CLASS_NAMES[cls_idx]:15s} -> {lesion_count} lesion regions flagged, saved: {out_path}")

print("\nDone. Lesion overlay images saved to images/")
