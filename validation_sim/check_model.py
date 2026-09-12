from pathlib import Path

import keras
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = (
    ROOT
    / "diabetic-retinopathy-detection"
    / "models"
)

MODEL_PATH = MODEL_DIR / "diabetic_retinopathy_model.keras"
SAMPLES_PATH = MODEL_DIR / "sample_images.npy"

CLASS_NAMES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR",
]


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file missing: {MODEL_PATH}"
        )

    print("Loading model...")

    # Inference only: training loss/optimizer are not required.
    model = keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    print("\nMODEL LOADED")
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)

    if not SAMPLES_PATH.exists():
        raise FileNotFoundError(
            f"Sample file missing: {SAMPLES_PATH}"
        )

    images = np.load(
        SAMPLES_PATH,
        allow_pickle=False,
    )

    print("\nSample array shape:", images.shape)

    if images.ndim != 4 or len(images) == 0:
        raise ValueError(
            "Expected a non-empty batch of images."
        )

    sample = images[:1].astype("float32")

    if not np.isfinite(sample).all():
        raise ValueError("Sample contains invalid pixel values.")

    print("Sample minimum:", float(sample.min()))
    print("Sample maximum:", float(sample.max()))

    # Saved sample arrays may already be normalized.
    # Do not divide by 255 again automatically.
    if sample.min() < 0 or sample.max() > 1:
        raise ValueError(
            "Sample is outside the documented 0–1 range. "
            "Verify sample preprocessing before proceeding."
        )

    expected_shape = tuple(model.input_shape[1:])

    if tuple(sample.shape[1:]) != expected_shape:
        raise ValueError(
            f"Sample shape {sample.shape[1:]} does not "
            f"match model input {expected_shape}."
        )

    predictions = np.asarray(
        model.predict(sample, verbose=0)
    )

    if predictions.shape != (1, 5):
        raise ValueError(
            f"Expected five class outputs, got {predictions.shape}"
        )

    scores = predictions[0]

    if not np.isfinite(scores).all():
        raise ValueError("Model returned invalid values.")

    if (
        np.any(scores < 0)
        or np.any(scores > 1)
        or not np.isclose(scores.sum(), 1.0, atol=1e-3)
    ):
        raise ValueError(
            "Outputs are not normalized probabilities. "
            "Check the model output layer."
        )

    grade = int(np.argmax(scores))

    print("\nPREDICTION CHECK")
    print("Predicted grade:", grade)
    print("Class:", CLASS_NAMES[grade])
    print("Raw confidence:", round(float(scores[grade]), 4))
    print("All class probabilities:", scores.tolist())

    print("\nModel loading and prediction check passed.")
    print("This is a smoke check, not cross-dataset validation.")


if __name__ == "__main__":
    main()