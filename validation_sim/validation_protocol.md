# Cross-Dataset Validation Protocol

## Overview
This document outlines the protocol used for the cross-dataset validation of the Diabetic Retinopathy screening pipeline (IQA + Classifier).

## Dataset Information
- **Dataset Source**: IDRiD (Indian Diabetic Retinopathy Image Dataset)
- **Dataset Split**: Testing Set
- **Label Source**: Official `IDRiD_Testing_Labels.csv` provided with the dataset
- **Evaluation Date**: 2026-09-11

## Pipeline Components
- **Classifier Checkpoint**: `diabetic_retinopathy_model.keras`
- **IQA Version**: Current implementation in `iqa_module/pipeline.py`
- **Baseline Preprocessing**: RGB mode, resized to 224x224, pixel values normalized to [0, 1] as defined by the model adapter.

### Dataset Independence & Leakage Risks

#### a) Checks Verified from Labels/Manifest
The evaluation manifest contains 103 IDRiD Testing records. All image IDs and DR grades match the supplied testing-label CSV (`IDRiD_Testing_Labels.csv`). All relative paths point to the `b. Testing Set` directory. No missing records, extra records, grade mismatches, duplicate sample IDs, or duplicate image paths were found. All 103 recorded SHA-256 hashes are well-formed and unique.

The supplied training labels contain 413 records (IDRiD_001 to IDRiD_413), and testing labels contain 103 records (IDRiD_001 to IDRiD_103). Image names are reused across splits.

#### b) Member 1's Reported Tuning History
According to the IQA module developer, initial threshold tuning used five images from the official IDRiD Training Set (IDRiD_001.jpg through IDRiD_005.jpg). A supplied provenance CSV confirms these were the only images used for tuning, and their five recorded hashes have no exact-file matches in the Testing manifest.

The developer also reports batch triage on 413 Training images: 356 PASS, 57 REJECT. Clarification was received that IQA thresholds were NOT changed after this batch evaluation.

A bug in the IQA module was discovered and fixed (commit `c884e85`): the missing-quadrant behaviour for illumination uniformity previously returned 1.0 when fewer than four quadrants contained retinal pixels. It now correctly returns 0.0, ensuring these images fail the uniformity threshold. Numeric thresholds otherwise stayed unchanged.

The IDRiD testing set has now been evaluated by Member 4. Any subsequent tuning informed by these results must be disclosed; the same test set must not then be described as untouched. We do not claim zero overall leakage or an untouched full pipeline.

#### c) Confirmed Leakage and Overlap
A full Training/Testing hash comparison of the actual local files verified one exact-file matching pair: Training `IDRiD_118.jpg` and Testing `IDRiD_064.jpg` share the same SHA-256 hash (`505330878f4d4186a71b8001570100b82b6b05cb431e81e2c259d8c584314708`). This actual-file hash verification was reported by Member 1 and verified locally. Due to this confirmed overlap, we cannot claim "zero leakage" or that the evaluation is "100% unbiased." 

The original 103-image evaluation and its original results are preserved. Any supplementary analysis excluding Testing `IDRiD_064.jpg` must be clearly labelled as post-hoc, must report its denominator, and must not replace the original results.

#### d) Evidence Still Unavailable
Full-pipeline (IQA + Classifier) independence on the IDRiD Testing set is not verified. It is possible that the IQA module has overfitted to the imaging characteristics of the IDRiD dataset.

The classifier is reported to have been trained on APTOS. The exact training and tuning manifests, as well as its calibration independence, remain unverified. Classifier training/calibration independence from IDRiD is still pending and has not been independently established.

### Conclusion
These results represent retrospective external-dataset evaluation of the existing classifier. We do not claim full clinical validation or deployment readiness based on this evaluation. The results reflect the isolated performance of the classifier on an external dataset, but the IQA performance may be optimistically biased due to potential dataset leakage during its tuning phase.

