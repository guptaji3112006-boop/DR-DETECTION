# Cross-Dataset Evaluation Summary

## Dataset Information
- **Dataset**: IDRiD Testing Set (Untouched for 0% data leakage)
- **Total Images**: 103
- **Referable DR Threshold**: Grade >= 2 (Moderate, Severe, Proliferative)

## Pipeline Workflow
Images were evaluated using the IQA Module for quality triage, followed by the frozen APTOS-trained model.

## IQA Rejection Rate
- **Accepted**: 94
- **Rejected**: 9
- **Rejection Rate**: 8.7%

## Metrics Summary

| Subset | Count | Sensitivity (Referable) | Specificity (Non-Referable) | Quadratic Weighted Kappa |
|---|---|---|---|---|
| All External Images | 103 | 0.9531 | 0.1282 | 0.1572 |
| IQA Accepted Images | 94 | 0.9667 | 0.1176 | 0.1694 |
| IQA Rejected Images | 9 | 0.75 | 0.2 | -0.1556 |

## Observations
- This evaluation correctly isolates model performance based on image quality as determined by the IQA pipeline.
- Model raw confidence is recorded in the granular predictions but is explicitly **not** reported as accuracy.
- The confusion matrix has been generated specifically for the clinically relevant workflow (IQA Accepted images).
