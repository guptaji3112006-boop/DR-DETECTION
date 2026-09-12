import unittest
import numpy as np
from iqa_module.quality_assessment import check_illumination_uniformity

class TestIQARegression(unittest.TestCase):
    def test_missing_quadrant_uniformity(self):
        # Create a dummy 100x100 image and mask
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        mask = np.zeros((100, 100), dtype=np.uint8)
        
        # Populate only 3 quadrants with valid mask pixels
        # Q1: top-left
        mask[10:40, 10:40] = 255
        # Q2: top-right
        mask[10:40, 60:90] = 255
        # Q3: bottom-left
        mask[60:90, 10:40] = 255
        # Q4 (bottom-right) is empty
        
        # The new fix should cause uniformity to be exactly 0.0 because there are <4 quadrants
        uniformity = check_illumination_uniformity(image, mask)
        self.assertEqual(uniformity, 0.0, f"Expected 0.0 for <4 quadrants, got {uniformity}")

if __name__ == "__main__":
    unittest.main()
