# DR Screening Validation Simulation

This module performs clinical validation of the Diabetic Retinopathy screening model (Feature 7) and simulates the screening workflow to handle realistic patient load (Feature 8).

## Implementation Status
- **Cross-Dataset Validation**: Implemented and executable.
- **Workflow Simulation**: Implemented and executable (SimPy).
- **Environment**: Reproducible `requirements.txt` unified for model and pipeline.
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
- **Gated**: Simulates the true clinical pipeline by running the IQA module first. If an image is flagged as `REJECT`, the classifier inference is skipped (`SKIPPED_BY_GATE`).

## Results and Limitations
- Outputs are saved in `validation_sim/results/`.
- Refer to `validation_protocol.md` for strict dataset independence limitations (IDRiD was partially used by Member 1 for IQA tuning, causing potential leakage).
- **Low Specificity Limitation**: The current frozen DR classifier demonstrates high sensitivity (>95%) but extremely poor specificity (~11%). It generates significant false positives. **This is a known limitation of the model, not a bug in the evaluation pipeline.**

## Simulation Assumptions
- A patient can have a maximum of 2 image capture attempts (one recapture if the first is rejected by IQA).
- Referral probability is simulated via static probabilities; they are decoupled from the AI predictions in the simulation context.