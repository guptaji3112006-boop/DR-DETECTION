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

## Dataset-Independence Claims & Leakage Risks

### Classifier Independence
The classifier was originally trained on the APTOS dataset. Therefore, evaluation on the IDRiD dataset represents an independent cross-dataset validation of the classifier's generalization capabilities.

### IQA Independence & Leakage Risk
The IQA module's thresholds were tuned using images from the IDRiD dataset. Specifically, `reports/iqa_triage_report.csv` indicates that IDRiD images were used for tuning. 
- A complete, hashed manifest of the exact images used for IQA tuning was not available.
- Due to the lack of verifiable tuning manifests, full-pipeline (IQA + Classifier) independence on the IDRiD Testing set is **not verified**. It is possible that the IQA module has overfitted to the imaging characteristics of the IDRiD dataset.

### Conclusion
We **do not claim** full clinical validation or deployment readiness based on this evaluation. The results reflect the isolated performance of the classifier on an external dataset, but the IQA performance may be optimistically biased due to potential dataset leakage during its tuning phase.
