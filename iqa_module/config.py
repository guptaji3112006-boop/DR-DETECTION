# iqa_module/config.py

# Standard image size for IQA and model input
IMAGE_SIZE = (512, 512)

SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png"]

# --- Clinical Thresholds (Empirically tuned on IDRiD) ---
# Retinal coverage within frame
MIN_FOV_COVERAGE = 0.50       # At least 50% must be valid retina

# Focus / Sharpness
MIN_SHARPNESS = 80.0          # Below this is unacceptable blur

# Illumination
MIN_BRIGHTNESS = 60.0         # Below this is underexposed / dark
MAX_BRIGHTNESS = 180.0        # Above this is washed-out / overexposed

# Detail / Contrast
MIN_CONTRAST = 12.0           # Below this lacks sufficient diagnostic detail

# Quality Score Gating
QUALITY_THRESHOLD = 0.60      # Composite score required to PASS