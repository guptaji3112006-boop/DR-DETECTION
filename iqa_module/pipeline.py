# iqa_module/pipeline.py
import os
import cv2
import numpy as np

try:
    from .config import IMAGE_SIZE
    from .quality_assessment import load_image, assess_quality, evaluate_quality, extract_fov_mask
    from .enhancement import enhance_retinal_image
except ImportError:
    from config import IMAGE_SIZE
    from quality_assessment import load_image, assess_quality, evaluate_quality, extract_fov_mask
    from enhancement import enhance_retinal_image

def process_retinal_image(image_input):
    """
    Frontline IQA & Retinal Enhancement Pipeline.
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found at: {image_input}")
        image = load_image(image_input)
    elif isinstance(image_input, np.ndarray):
        image = cv2.resize(image_input, IMAGE_SIZE)
    else:
        raise TypeError("Input must be a valid filepath string or numpy array.")

    metrics = assess_quality(image)
    eval_result = evaluate_quality(metrics)

    is_usable = (eval_result["status"] == "PASS")

    if is_usable:
        mask = extract_fov_mask(image)
        enhanced_image = enhance_retinal_image(image, mask=mask)
    else:
        enhanced_image = None

    return {
        "status": eval_result["status"],
        "is_usable": is_usable,
        "quality_score": eval_result["quality_score"],
        "reason": eval_result["reason"],
        "clinical_guidance": eval_result.get("clinical_guidance", eval_result.get("reason", "Suboptimal capture; retake recommended.")),
        "metrics": metrics,
        "processed_image": enhanced_image,
        "enhanced_image": enhanced_image,
    }