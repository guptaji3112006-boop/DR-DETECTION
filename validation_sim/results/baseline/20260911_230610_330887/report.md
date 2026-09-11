# Cross-Dataset Evaluation Report

## Run Information

- Run ID: 20260911_230610_330887
- Mode: baseline
- Checkpoint SHA-256: 52058ebf6d7a58d940d6b1ffcc7b00c0574a7153cdc5769ad3cd48ca84759b3f
- Dataset independence: not independently verified.

## Execution Counts

- **manifest_total**: 103
- **attempted**: 103
- **input_failures**: 0
- **valid_predictions**: 103
- **inference_attempted**: 103
- **inference_failures**: 0
- **skipped_by_gate**: 0
- **iqa_pass**: 94
- **iqa_reject**: 9
- **iqa_error**: 0
- **iqa_not_run**: 0
- **valid_accepted**: 94

## Coverage and Rejections

- IQA rejection rate among 103 successful IQA assessments:
  8.7%
- IQA accepted images with valid predictions / all attempted:
  91.3%

## Metrics

Binary target: severity grade >= 2.
This is a grade-based referable-DR proxy without a separate
macular-edema assessment.

| Subset | Count | Sensitivity | Specificity | Weighted Kappa |
|---|---:|---:|---:|---:|
| All Successfully Predicted External Images | 103 | 0.9531 | 0.1282 | 0.1572 |
| IQA Accepted Images (Valid Predictions) | 94 | 0.9667 | 0.1176 | 0.1694 |
| IQA Rejected Images (Offline Analysis Only) | 9 | 0.75 | 0.2 | -0.1556 |

## Interpretation

- Baseline mode predicts on original images and groups results by IQA.
- Rejected-image predictions in baseline mode are offline analysis only.
- Gated mode skips inference unless IQA explicitly returns PASS.
- Accepted images use the original-image classifier preprocessing.
  This run does not evaluate enhanced-image classifier input.
- Raw confidence is not accuracy and is not calibrated.
- Sensitivity must be considered alongside specificity and coverage.
- Accepted-subset metrics alone do not prove that IQA improves performance.

## Confidence Intervals

The metrics CSV includes 95% Wilson score intervals for sensitivity
and specificity. Intervals are image-level and assume independent
observations. Patient clustering, if present, is not accounted for.

## Limitations

The checkpoint is reported as APTOS-trained. Exact training overlap
has not been independently verified. IQA thresholds were developed
using IDRiD data; full-pipeline independence remains unverified.

This is retrospective dataset evaluation, not proof of clinical
validation or deployment readiness.
