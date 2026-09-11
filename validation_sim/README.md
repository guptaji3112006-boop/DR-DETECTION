# Validation and Screening Simulation

Member 4 module for the DR Detection project.

## Responsibilities

- Evaluate the model on independent labeled datasets.
- Simulate patient arrivals, image capture, IQA recapture,
  AI processing, doctor review, and referral routing.
- Compare screening capacity and waiting times across scenarios.

## Folders

- config/: Simulation settings.
- results/: Generated metrics, CSV files, and charts.

## Status

Screening simulation, scenario comparison, CSV exports, and
comparison charts are implemented. Cross-dataset model
validation is pending.

## Run the simulation

Run from the DR-DETECTION repository root:

    python -m validation_sim.simpy_model
    python -m validation_sim.run_scenarios
    python -m validation_sim.plot_results

## Scenarios

| Scenario | Patients/hour | Cameras | Reviewers |
|---|---:|---:|---:|
| Baseline | 10 | 1 | 1 |
| High patient load | 20 | 1 | 1 |
| Extra camera | 20 | 2 | 1 |

Each scenario runs with five random seeds.

## Generated outputs

- results/scenario_runs.csv: Results for individual runs.
- results/scenario_results.csv: Scenario means and standard deviations.
- results/patient_records.csv: Simulated patient records at closing.
- results/scenario_comparison.png: Scenario comparison charts.

## Assumptions

- The screening day lasts 480 minutes.
- Patient arrival intervals follow an exponential distribution.
- Capture, IQA, AI processing, and review durations are fixed.
- The camera remains occupied during IQA.
- Each patient has at most two capture attempts.
- Repeated IQA rejection routes to manual review without AI grading.
- Every patient receives doctor review.
- Referral uses an assumed probability, not a clinical prediction.
- Processing stops at closing; unfinished patients form the backlog.

## Interpretation

Waiting-time averages include completed patients only.
They must be interpreted alongside the closing backlog.

Error bars show standard deviation across simulation runs,
not confidence intervals.

## Limitations

The timings and probabilities are illustrative assumptions.
The simulation does not establish clinical accuracy or prove
real-world district capacity.

Specialist appointment queues and treatment are not modeled.

Cross-dataset model validation is pending.

## Simulation Results

Results below are averages across five runs.
All scenarios simulate an eight-hour working day.

| Scenario | Mean arrivals | Mean completed | Mean pending at close | Mean queue wait (minutes) |
|---|---:|---:|---:|---:|
| Baseline | 82.4 | 80.6 | 1.8 | 2.89 |
| High patient load | 155.8 | 129.0 | 26.8 | 48.51 |
| Extra camera | 162.0 | 158.8 | 3.2 | 2.33 |

## Findings

- Under baseline conditions, an average of 80.6 patients
  completed screening, with 1.8 patients pending at closing.

- Increasing the configured arrival rate from 10 to 20 patients
  per hour increased the average closing backlog to 26.8 patients.
  Average queue wait among completed patients rose to 48.51 minutes.

- With two cameras at the same configured high arrival rate,
  average completed screenings reached 158.8 and the closing
  backlog fell to 3.2 patients. Average queue wait among completed
  patients was 2.33 minutes.

- These results suggest that camera capacity is an important
  bottleneck under the current simulation assumptions.

## Interpretation Notes

- Arrival counts vary because arrivals are randomly simulated.
  The high-load scenarios have the same configured arrival rate,
  but their realized patient counts are different.

- Waiting-time averages include completed patients only.
  Unfinished patients are reported separately as the closing backlog.

- These are simulated operational results, not clinical
  model-accuracy measurements or verified hospital performance.