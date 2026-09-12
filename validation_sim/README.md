# DR Screening Validation Simulation

This module performs retrospective external-dataset evaluation of the
existing DR classifier on IDRiD and SimPy simulation of a screening workflow.
These results do not establish clinical validation or deployment readiness.

## Implementation Status
- **Cross-Dataset Validation**: Implemented and executable.
- **Workflow Simulation**: Implemented and executable (SimPy).
- **Environment**: Reproducible `requirements.txt` unified for model and pipeline (tested on Python 3.11).
- **Dependencies on other members**: 
  - Depends on Member 1's IQA Module for triage thresholds.
  - Depends on Member 2's frozen APTOS-trained DR model.
  - Handoff ready for UI Dashboard integration.

## Environment Setup
Run from the root of the repository:
```powershell
python -m venv .venv-model
.venv-model\Scripts\python.exe -m pip install -r validation_sim/requirements.txt
```

## Running the Pipeline
Run the full suite of tests, data validation, metrics generation, and simulations:
```powershell
.\validation_sim\run_pipeline.ps1
```

## Dataset Arrangement
The cross-dataset validation relies on the external `IDRiD` testing set.
Place it at: `external_data/idrid/images/idrid_full_dataset/idrid/B. Disease Grading/1. Original Images/b. Testing Set`.
The label manifest should be at: `external_data/idrid/IDRiD_Testing_Labels.csv`.

## Evaluation Modes (Baseline vs. Gated)
- **Baseline**: Passes all test images through the raw DR Model and computes metrics for the whole set, while also silently recording their IQA status.
- **Gated**: Runs IQA before classification. Rejected images are skipped.
  Accepted images are classified using the current model adapter.
  This experiment evaluates gating with the existing classifier; it does
  not establish validation of the complete planned lesion-aware pipeline.

## Results and Limitations
- Outputs are saved in `validation_sim/results/`.
- Refer to `validation_protocol.md` for strict dataset independence limitations (IDRiD was partially used by Member 1 for IQA tuning, causing potential leakage. One cross-split matching hash documented).
- Initial tuning provenance received; classifier training/calibration independence pending.
- IQA fix (commit c884e85) is a behaviour change after the earlier evaluation, addressing missing-quadrant uniformity edge-cases.

### Recorded IDRiD Results

| Evaluation subset | Graded images | Sensitivity | Specificity | QWK |
|---|---:|---:|---:|---:|
| Baseline: all successfully predicted images | 103 | 0.9531 | 0.1282 | 0.1572 |
| IQA-accepted images / gated predictions | 94 | 0.9667 | 0.1176 | 0.1694 |

The gated run skipped 9 of 103 images: coverage was 91.26%.
Skipped images are abstentions, not negative predictions.

Sensitivity and specificity use grade >= 2 as the referable-DR proxy;
this does not incorporate a separate diabetic macular edema assessment.

Baseline and gated metrics use different evaluated populations.
Their difference alone does not demonstrate improved model performance.
Low specificity indicates many false-positive predictions.

- **Low Specificity Limitation**: The current frozen DR classifier demonstrates high sensitivity (>95%) but extremely poor specificity (~11%). It generates significant false positives. We must still verify preprocessing logic, class mapping, and training provenance before concluding this cannot be caused by an implementation issue.
- **Uncalibrated Confidence**: The raw confidence output by the model is uncalibrated and should not be interpreted as a true probability.
- **Accepted-subset metrics**: Accepted-subset metrics alone do not prove an accuracy improvement.

## Simulation Assumptions
- A patient can have a maximum of 2 image capture attempts (one recapture if the first is rejected by IQA). Note: this actual configured capture-attempt limit is a mismatch with Member 1's recapture documentation and is pending team agreement.
- Referral probability is simulated via static probabilities; they are decoupled from the AI predictions in the simulation context.
- `patient_records.csv` contains simulated patient summaries.
- `referrals_completed` currently counts simulated referral decisions among completed screenings, not completed specialist visits.

## Latest Simulation Results

Averages across five runs of an eight-hour screening day.

| Scenario | Mean completed | Mean pending at close | Mean queue wait (minutes) |
|---|---:|---:|---:|
| Baseline | 85.8 | 1.4 | 4.34 |
| High patient load | 129.6 | 35.8 | 57.86 |
| Extra camera | 162.2 | 3.2 | 2.80 |

Queue-wait averages include completed patients only.
These results use simulated arrivals and assumed service times.