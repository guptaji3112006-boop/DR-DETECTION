import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
import pandas as pd
import subprocess
import sys


class TestAuditOverlap(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.train_dir = self.temp_dir / "train_images"
        self.test_dir = self.temp_dir / "test_images"
        self.out_dir = self.temp_dir / "output"
        self.train_dir.mkdir()
        self.test_dir.mkdir()
        self.out_dir.mkdir()
        
        self.train_labels = self.temp_dir / "train_labels.csv"
        self.test_labels = self.temp_dir / "test_labels.csv"
        self.script_path = Path("validation_sim/audit_split_overlap.py").resolve()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def run_audit(self):
        cmd = [
            sys.executable, str(self.script_path),
            "--train-labels", str(self.train_labels),
            "--train-image-dir", str(self.train_dir),
            "--test-labels", str(self.test_labels),
            "--test-image-dir", str(self.test_dir),
            "--output-dir", str(self.out_dir)
        ]
        return subprocess.run(cmd, capture_output=True, text=True)

    def test_same_basename_different_bytes(self):
        (self.train_dir / "img1.jpg").write_bytes(b"DATA_A")
        (self.test_dir / "img1.jpg").write_bytes(b"DATA_B")
        
        pd.DataFrame([{"Image name": "img1", "Retinopathy grade": 1}]).to_csv(self.train_labels, index=False)
        pd.DataFrame([{"Image name": "img1", "Retinopathy grade": 2}]).to_csv(self.test_labels, index=False)
        
        res = self.run_audit()
        self.assertEqual(res.returncode, 0, res.stderr)
        
        out_dirs = list(self.out_dir.glob("audit_overlap_*"))
        self.assertEqual(len(out_dirs), 1)
        summary = json.loads((out_dirs[0] / "summary.json").read_text())
        self.assertEqual(summary["exact_duplicates_found"], 0)

    def test_different_basename_identical_bytes(self):
        (self.train_dir / "train_img.jpg").write_bytes(b"IDENTICAL_DATA")
        (self.test_dir / "test_img.png").write_bytes(b"IDENTICAL_DATA")
        
        pd.DataFrame([{"Image name": "train_img", "Retinopathy grade": 1}]).to_csv(self.train_labels, index=False)
        pd.DataFrame([{"Image name": "test_img", "Retinopathy grade": 1}]).to_csv(self.test_labels, index=False)
        
        res = self.run_audit()
        self.assertNotEqual(res.returncode, 0, "Should fail (nonzero exit) when overlap is found")
        
        out_dirs = list(self.out_dir.glob("audit_overlap_*"))
        summary = json.loads((out_dirs[0] / "summary.json").read_text())
        self.assertEqual(summary["exact_duplicates_found"], 1)

    def test_missing_image_fails_clearly(self):
        (self.train_dir / "exists.jpg").write_bytes(b"DATA")
        
        pd.DataFrame([{"Image name": "exists", "Retinopathy grade": 1}, 
                      {"Image name": "missing", "Retinopathy grade": 2}]).to_csv(self.train_labels, index=False)
        pd.DataFrame([{"Image name": "test", "Retinopathy grade": 1}]).to_csv(self.test_labels, index=False)
        (self.test_dir / "test.jpg").write_bytes(b"TEST")
        
        res = self.run_audit()
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Image missing for missing", res.stderr)

    def test_duplicate_label_ids_fail_clearly(self):
        (self.train_dir / "img.jpg").write_bytes(b"DATA")
        pd.DataFrame([{"Image name": "img", "Retinopathy grade": 1}, 
                      {"Image name": "img", "Retinopathy grade": 2}]).to_csv(self.train_labels, index=False)
        pd.DataFrame([{"Image name": "test", "Retinopathy grade": 1}]).to_csv(self.test_labels, index=False)
        (self.test_dir / "test.jpg").write_bytes(b"TEST")
        
        res = self.run_audit()
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Duplicate label IDs found", res.stderr)

    def test_ambiguous_matching_files_fail_clearly(self):
        (self.train_dir / "img.jpg").write_bytes(b"DATA1")
        (self.train_dir / "img.png").write_bytes(b"DATA2")
        
        pd.DataFrame([{"Image name": "img", "Retinopathy grade": 1}]).to_csv(self.train_labels, index=False)
        pd.DataFrame([{"Image name": "test", "Retinopathy grade": 1}]).to_csv(self.test_labels, index=False)
        (self.test_dir / "test.jpg").write_bytes(b"TEST")
        
        res = self.run_audit()
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Ambiguous image match for img", res.stderr)

if __name__ == "__main__":
    unittest.main()
