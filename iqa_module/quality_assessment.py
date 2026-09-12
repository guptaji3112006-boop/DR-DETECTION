# iqa_module/quality_assessment.py
import cv2
import numpy as np

try:
    from .config import (
        IMAGE_SIZE,
        MIN_FOV_COVERAGE,
        MIN_SHARPNESS,
        MIN_BRIGHTNESS,
        MAX_BRIGHTNESS,
        MIN_CONTRAST,
        QUALITY_THRESHOLD,
    )
except ImportError:
    from config import (
        IMAGE_SIZE,
        MIN_FOV_COVERAGE,
        MIN_SHARPNESS,
        MIN_BRIGHTNESS,
        MAX_BRIGHTNESS,
        MIN_CONTRAST,
        QUALITY_THRESHOLD,
    )


def load_image(image_path):
    """
    Load a retinal image from disk and resize to standard dimensions.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from: {image_path}")

    # Standardize size for consistent metric scale
    image = cv2.resize(image, IMAGE_SIZE)
    return image


def extract_fov_mask(image, threshold=10):
    """
    Generate a binary mask isolating the circular retinal Field of View (FOV)
    from the black surrounding background.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Pixels brighter than threshold are considered retina
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Fill small holes inside the mask (e.g. dark blood vessels)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def calculate_sharpness(image, mask=None):
    """
    Measure focus/sharpness using Laplacian variance inside the retinal region.
    Higher value -> sharper image; Lower value -> blurry image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    if mask is not None:
        retina_laplacian = laplacian[mask > 0]
        if len(retina_laplacian) == 0:
            return 0.0
        return float(retina_laplacian.var())

    return float(laplacian.var())


def calculate_brightness(image, mask=None):
    """
    Calculate average brightness strictly within the retinal tissue.
    Range: 0 (pitch black) to 255 (pure white).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if mask is not None:
        retina_pixels = gray[mask > 0]
        if len(retina_pixels) == 0:
            return 0.0
        return float(retina_pixels.mean())

    return float(gray.mean())


def calculate_contrast(image, mask=None):
    """
    Calculate contrast (standard deviation) strictly within the retinal tissue.
    Higher value -> better dynamic range / clearer details.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if mask is not None:
        retina_pixels = gray[mask > 0]
        if len(retina_pixels) == 0:
            return 0.0
        return float(retina_pixels.std())

    return float(gray.std())


def check_illumination_uniformity(image, mask):
    """
    Check if illumination is balanced across all 4 quadrants of the retina.
    Detects pupil shadowing and lateral camera misalignment.
    Score: 0.0 (severely shadowed on one side) to 1.0 (perfectly even).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    mid_y, mid_x = h // 2, w // 2

    # Split into 4 quadrants: Top-Left, Top-Right, Bottom-Left, Bottom-Right
    quadrants = [
        (gray[:mid_y, :mid_x], mask[:mid_y, :mid_x]),
        (gray[:mid_y, mid_x:], mask[:mid_y, mid_x:]),
        (gray[mid_y:, :mid_x], mask[mid_y:, :mid_x]),
        (gray[mid_y:, mid_x:], mask[mid_y:, mid_x:]),
    ]

    means = []
    for q_img, q_mask in quadrants:
        pts = q_img[q_mask > 0]
        if len(pts) > 0:
            means.append(float(pts.mean()))

    if len(means) < 4:
        # Edge case: If retina is completely absent in one or more quadrants (e.g., severe
        # decentration or crescent shadow), illumination is by definition non-uniform.
        # Returning 0.0 ensures the image fails the uniformity threshold (<0.60) instead
        # of receiving an artificially perfect score.
        return 0.0

    # Ratio of darkest quadrant to brightest quadrant
    uniformity = min(means) / (max(means) + 1e-5)
    return float(np.round(uniformity, 3))


def detect_specular_artifacts(image, mask, glare_threshold=250):
    """
    Detect flash glare or over-saturated corneal reflections (>250 intensity).
    Returns ratio of glare pixels to total retinal pixels.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glare_pixels = (gray >= glare_threshold) & (mask > 0)
    retina_pixels = (mask > 0).sum()

    if retina_pixels == 0:
        return 0.0

    glare_ratio = float(glare_pixels.sum()) / float(retina_pixels)
    return float(np.round(glare_ratio, 4))


def assess_quality(image):
    """
    Compute full clinical quality metrics inside the retinal Field of View (FOV).
    """
    mask = extract_fov_mask(image)

    return {
        "sharpness": calculate_sharpness(image, mask),
        "brightness": calculate_brightness(image, mask),
        "contrast": calculate_contrast(image, mask),
        "fov_coverage": float((mask > 0).mean()),
        "uniformity": check_illumination_uniformity(image, mask),
        "glare_ratio": detect_specular_artifacts(image, mask),
    }


def calculate_quality_score(metrics):
    """
    Compute a composite, normalized quality score between 0.0 and 1.0.
    Weights: Sharpness (35%), Brightness (25%), Contrast (25%), Uniformity (15%).
    """
    # Normalize Sharpness: benchmark 200+ as ideal (1.0)
    sharp_norm = np.clip(metrics["sharpness"] / 200.0, 0.0, 1.0)

    # Normalize Brightness: ideal range is roughly 90-110
    bright = metrics["brightness"]
    if 90 <= bright <= 120:
        bright_norm = 1.0
    elif bright < 90:
        bright_norm = np.clip((bright - MIN_BRIGHTNESS) / (90 - MIN_BRIGHTNESS), 0.0, 1.0)
    else:
        bright_norm = np.clip((MAX_BRIGHTNESS - bright) / (MAX_BRIGHTNESS - 120), 0.0, 1.0)

    # Normalize Contrast: benchmark 25+ as ideal (1.0)
    contrast_norm = np.clip(metrics["contrast"] / 25.0, 0.0, 1.0)

    # Normalize Uniformity: 1.0 is ideal
    uniformity_norm = np.clip(metrics["uniformity"], 0.0, 1.0)

    score = (
        (0.35 * sharp_norm)
        + (0.25 * bright_norm)
        + (0.25 * contrast_norm)
        + (0.15 * uniformity_norm)
    )

    # Penalize if severe glare covers >1.5% of retina
    if metrics["glare_ratio"] > 0.015:
        score *= 0.80

    return float(np.round(score, 3))


def evaluate_quality(metrics):
    """
    Determine PASS / REJECT status with actionable clinical guidance for screening workers.
    """
    reasons = []
    clinical_guidance = []

    # 1. Check hard failure thresholds
    if metrics["fov_coverage"] < MIN_FOV_COVERAGE:
        reasons.append("Insufficient retinal coverage (<50% retina visible)")
        clinical_guidance.append("Re-center camera over the pupil; ensure eye is wide open.")

    if metrics["sharpness"] < MIN_SHARPNESS:
        reasons.append(f"Blur detected ({metrics['sharpness']:.1f} < {MIN_SHARPNESS})")
        clinical_guidance.append("Patient moved or camera out-of-focus; hold camera steady and refocus.")

    if metrics["brightness"] < MIN_BRIGHTNESS:
        reasons.append(f"Image underexposed ({metrics['brightness']:.1f} < {MIN_BRIGHTNESS})")
        clinical_guidance.append("Lighting too dark; increase camera flash intensity or check pupil dilation.")
    elif metrics["brightness"] > MAX_BRIGHTNESS:
        reasons.append(f"Image overexposed ({metrics['brightness']:.1f} > {MAX_BRIGHTNESS})")
        clinical_guidance.append("Flash glare or too bright; reduce flash setting.")

    if metrics["contrast"] < MIN_CONTRAST:
        reasons.append(f"Low tissue contrast ({metrics['contrast']:.1f} < {MIN_CONTRAST})")
        clinical_guidance.append("Possible media opacity (cataract) or dirty lens; verify lens cleanliness.")

    if metrics["uniformity"] < 0.60:
        reasons.append(f"Severe uneven illumination (uniformity: {metrics['uniformity']:.2f} < 0.60)")
        clinical_guidance.append("Pupil shadow detected; realign camera perpendicularly with pupil axis.")

    if metrics["glare_ratio"] > 0.02:
        reasons.append(f"Corneal flash reflection detected ({metrics['glare_ratio']*100:.1f}% glare)")
        clinical_guidance.append("Corneal reflection artifact; tilt camera slightly to shift reflection away from macula.")

    # 2. Compute overall composite score
    score = calculate_quality_score(metrics)

    # 3. Final Decision Logic
    if len(reasons) == 0 and score >= QUALITY_THRESHOLD:
        status = "PASS"
        decision_reason = "Good clinical quality; suitable for DR grading."
        guidance = "Image accepted. Proceeding to diagnostic classification."
    else:
        status = "REJECT"
        # If no hard failure was triggered, explain why the composite score is low
        if not reasons:
            weaknesses = []
            if metrics["sharpness"] < 130.0:
                weaknesses.append("marginal sharpness")
                clinical_guidance.append("Hold camera steady and ensure patient fixates on target.")
            if metrics["brightness"] < 85.0:
                weaknesses.append("suboptimal illumination")
                clinical_guidance.append("Increase illumination slightly or check pupil alignment.")
            if metrics["contrast"] < 18.0:
                weaknesses.append("low tissue contrast")
                clinical_guidance.append("Clean lens and verify focus.")

            detail = ", ".join(weaknesses) if weaknesses else "overall low diagnostic signal"
            reasons.append(f"Composite quality score ({score:.2f} < {QUALITY_THRESHOLD}) due to {detail}")

        decision_reason = "; ".join(reasons)
        guidance = " | ".join(clinical_guidance) if clinical_guidance else "Capture quality suboptimal; please retake the photograph."

    return {
        "status": status,
        "quality_score": score,
        "reason": decision_reason,
        "clinical_guidance": guidance,
        "metrics": metrics,
    }