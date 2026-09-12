import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from validation_sim.evaluate import calculate_metrics, is_referable
from validation_sim.simpy_model import validate_settings

class TestEvaluationMetrics(unittest.TestCase):
    def test_calculate_metrics_perfect(self):
        df = pd.DataFrame({
            "true_grade": [0, 1, 2, 3, 4],
            "predicted_grade": [0, 1, 2, 3, 4]
        })
        m = calculate_metrics(df, "Test")
        self.assertEqual(m["sensitivity"], 1.0)
        self.assertEqual(m["specificity"], 1.0)
        self.assertEqual(m["qwk"], 1.0)
        self.assertEqual(m["tp"], 3) # 2, 3, 4
        self.assertEqual(m["tn"], 2) # 0, 1

    def test_calculate_metrics_undefined(self):
        df = pd.DataFrame({
            "true_grade": [2, 3, 4], # Only positives
            "predicted_grade": [2, 3, 4]
        })
        m = calculate_metrics(df, "Test")
        self.assertEqual(m["sensitivity"], 1.0)
        self.assertEqual(m["specificity"], "N/A")
        
    def test_is_referable(self):
        self.assertEqual(is_referable(0), 0)
        self.assertEqual(is_referable(1), 0)
        self.assertEqual(is_referable(2), 1)
        self.assertEqual(is_referable(3), 1)
        self.assertEqual(is_referable(4), 1)

class TestSimulationValidation(unittest.TestCase):
    def test_validate_settings_valid(self):
        settings = {
            "patients_per_hour": 10,
            "cameras": 1,
            "ai_workers": 1,
            "reviewers": 1,
            "max_capture_attempts": 2,
            "capture_time_minutes": 2,
            "iqa_time_minutes": 1,
            "ai_time_minutes": 1,
            "review_time_minutes": 5,
            "iqa_rejection_probability": 0.1,
            "referral_probability": 0.2
        }
        # Should not raise exception
        validate_settings(settings, 480)

    def test_validate_settings_invalid(self):
        settings = {
            "patients_per_hour": -10,
        }
        with self.assertRaises(ValueError):
            validate_settings(settings, 480)
            
if __name__ == '__main__':
    unittest.main()
