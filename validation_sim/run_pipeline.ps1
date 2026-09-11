$ErrorActionPreference = "Stop"

$env:PYTHONPATH = (Get-Location).Path

Write-Host "Running Tests..."
.\.venv-model\Scripts\python.exe -m unittest validation_sim/test_pipeline.py

Write-Host "Generating Manifest..."
.\.venv-model\Scripts\python.exe validation_sim/prepare_manifest.py `
    --labels "external_data/idrid/IDRiD_Testing_Labels.csv" `
    --image-dir "external_data/idrid/images/idrid_full_dataset/idrid/B. Disease Grading/1. Original Images/b. Testing Set" `
    --dataset-name "IDRiD" `
    --split "Testing" `
    --dataset-root "external_data" `
    --output "validation_sim/results/test_manifest.csv"

Write-Host "Running Baseline Evaluation..."
.\.venv-model\Scripts\python.exe validation_sim/evaluate.py `
    --manifest "validation_sim/results/test_manifest.csv" `
    --dataset-root "external_data" `
    --checkpoint "diabetic-retinopathy-detection/models/diabetic_retinopathy_model.keras" `
    --output-dir "validation_sim/results/baseline" `
    --mode "baseline"

Write-Host "Running Gated Evaluation..."
.\.venv-model\Scripts\python.exe validation_sim/evaluate.py `
    --manifest "validation_sim/results/test_manifest.csv" `
    --dataset-root "external_data" `
    --checkpoint "diabetic-retinopathy-detection/models/diabetic_retinopathy_model.keras" `
    --output-dir "validation_sim/results/gated" `
    --mode "gated"

Write-Host "Running Simulation Scenarios..."
.\.venv-model\Scripts\python.exe validation_sim/run_scenarios.py

Write-Host "Plotting Results..."
.\.venv-model\Scripts\python.exe validation_sim/plot_results.py

Write-Host "Pipeline execution completed successfully."
