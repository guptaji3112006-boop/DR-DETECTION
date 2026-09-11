# iqa_module/enhancement.py
import cv2
import numpy as np
try:
    from .quality_assessment import extract_fov_mask
except ImportError:
    from quality_assessment import extract_fov_mask


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    in the LAB color space to enhance lesion visibility without color distortion.
    """
    # Convert BGR to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE specifically to the Lightness channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_l = clahe.apply(l_channel)

    # Merge channels back and convert back to BGR
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced_bgr


def enhance_retinal_image(image, mask=None):
    """
    Full enhancement pipeline for fundus images:
    1. Extracts FOV mask (if not provided).
    2. Applies LAB-CLAHE contrast enhancement.
    3. Masks out background to ensure clean black borders.
    """
    if mask is None:
        mask = extract_fov_mask(image)

    # 1. Enhance contrast and local illumination
    enhanced = apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8))

    # 2. Preserve clean black background outside FOV
    clean_enhanced = cv2.bitwise_and(enhanced, enhanced, mask=mask)

    return clean_enhanced