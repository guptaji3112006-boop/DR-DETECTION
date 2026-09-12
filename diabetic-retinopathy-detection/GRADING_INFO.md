# Grading Module Information

## Final Model
- Model: diabetic_retinopathy_model.keras
- Location: models/diabetic_retinopathy_model.keras

## Inference / Preprocessing
- Code: app.py
- Image size: 224 x 224
- Image format: RGB
- Pixel normalization: 0 to 1

## Training Code
- Code: scripts/train_model.py
- Set `DR_TRAIN_IMAGES_DIR` to the APTOS `train_images` directory before running it.
- Deterministic manifests are written to `models/*_split.csv`.

## Calibration
- Fit on the reserved calibration manifest with `python scripts/fit_calibration.py --images <APTOS train_images>`.
- Output: `models/calibration.json`.

## Training Dataset
- Dataset: APTOS 2019 Blindness Detection
- Number of classes: 5

## Classes
1. No DR
2. Mild DR
3. Moderate DR
4. Severe DR
5. Proliferative DR

## Scope Note
The repository contains a trained five-class baseline and a lesion-aware inference contract. It does not contain a trained lesion segmentation checkpoint or lesion-labelled second-stage severity head; those must be supplied by the upstream segmentation owner before claiming a fully trained two-stage model.