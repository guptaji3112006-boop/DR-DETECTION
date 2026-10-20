import numpy as np
import cv2
from app import detect_lesions
import json

# create a dummy image (e.g. 500x500 RGB) with some bright and dark spots
img = np.zeros((500, 500, 3), dtype=np.uint8)

# Make FOV visible (gray)
cv2.circle(img, (250, 250), 240, (50, 50, 50), -1)

# Add a dark spot (simulating hemorrhage)
cv2.circle(img, (200, 200), 10, (10, 10, 10), -1)

# Add a bright spot (simulating exudate)
cv2.circle(img, (300, 300), 12, (200, 200, 200), -1)

# convert to float32 [0, 1] as expected by the detector
img_float = img.astype(np.float32) / 255.0

overlay, breakdown = detect_lesions(img_float)

print(json.dumps(breakdown, indent=2))
