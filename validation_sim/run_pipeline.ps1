$ErrorActionPreference = "Stop"

# Locate the repository from this script's location.
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv-model\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw "Python environment missing: $pythonExe"
}

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory = $true)]
        [string]$StepName,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host "=== $StepName ===" -ForegroundColor Cyan

    & $pythonExe @Arguments

    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed with exit code $LASTEXITCODE. Pipeline stopped."
    }

    Write-Host "$StepName passed." -ForegroundColor Green
}

Push-Location $repoRoot

try {
    Invoke-PythonStep -StepName "Checking dependencies" -Arguments @(
        "-m"
        "pip"
        "check"
    )

    Invoke-PythonStep -StepName "Checking Python syntax" -Arguments @(
        "-m"
        "py_compile"
        "validation_sim/evaluate.py"
        "validation_sim/prepare_manifest.py"
        "validation_sim/model_adapter.py"
        "validation_sim/simpy_model.py"
        "validation_sim/run_scenarios.py"
        "validation_sim/plot_results.py"
    )

    Invoke-PythonStep -StepName "Running tests" -Arguments @(
        "-m"
        "unittest"
        "validation_sim.test_pipeline"
        "-v"
    )

    Invoke-PythonStep -StepName "Generating manifest" -Arguments @(
        "-m"
        "validation_sim.prepare_manifest"
        "--labels"
        "external_data/idrid/IDRiD_Testing_Labels.csv"
        "--image-dir"
        "external_data/idrid/images/idrid_full_dataset/idrid/B. Disease Grading/1. Original Images/b. Testing Set"
        "--dataset-name"
        "IDRiD"
        "--split"
        "Testing"
        "--dataset-root"
        "external_data"
        "--output"
        "validation_sim/results/test_manifest.csv"
    )

    Invoke-PythonStep -StepName "Running baseline evaluation" -Arguments @(
        "-m"
        "validation_sim.evaluate"
        "--manifest"
        "validation_sim/results/test_manifest.csv"
        "--dataset-root"
        "external_data"
        "--checkpoint"
        "diabetic-retinopathy-detection/models/diabetic_retinopathy_model.keras"
        "--output-dir"
        "validation_sim/results/baseline"
        "--mode"
        "baseline"
    )

    Invoke-PythonStep -StepName "Running gated evaluation" -Arguments @(
        "-m"
        "validation_sim.evaluate"
        "--manifest"
        "validation_sim/results/test_manifest.csv"
        "--dataset-root"
        "external_data"
        "--checkpoint"
        "diabetic-retinopathy-detection/models/diabetic_retinopathy_model.keras"
        "--output-dir"
        "validation_sim/results/gated"
        "--mode"
        "gated"
    )

    Invoke-PythonStep -StepName "Running simulation scenarios" -Arguments @(
        "-m"
        "validation_sim.run_scenarios"
    )

    Invoke-PythonStep -StepName "Plotting results" -Arguments @(
        "-m"
        "validation_sim.plot_results"
    )

    Write-Host ""
    Write-Host "Pipeline execution completed successfully." -ForegroundColor Green
    Write-Host "Results: validation_sim/results/"
}
finally {
    Pop-Location
}