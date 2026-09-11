import argparse
import json
from pathlib import Path

import keras
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = (
    ROOT
    / "diabetic-retinopathy-detection"
    / "models"
    / "diabetic_retinopathy_model.keras"
)

CLASS_NAMES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR",
]


class DRModelAdapter:
    def __init__(self, model_path=DEFAULT_MODEL):
        self.model_path = Path(model_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        self.model = keras.models.load_model(
            self.model_path,
            compile=False,
        )

        if tuple(self.model.input_shape[1:]) != (224, 224, 3):
            raise ValueError("Expected model input: 224 x 224 x 3")

        if self.model.output_shape[-1] != 5:
            raise ValueError("Expected five output classes.")

    @staticmethod
    def preprocess(image_path):
        image_path = Path(image_path)

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        # Match the existing app's resize-then-RGB preprocessing.
        with Image.open(image_path) as image:
            image = image.resize((224, 224)).convert("RGB")
            array = np.asarray(image, dtype=np.float32) / 255.0

        return np.expand_dims(array, axis=0)

    def predict(self, image_path):
        model_input = self.preprocess(image_path)

        output = np.asarray(
            self.model.predict(model_input, verbose=0)
        )

        if output.shape != (1, 5):
            raise ValueError(
                f"Unexpected prediction shape: {output.shape}"
            )

        probabilities = output[0]

        if (
            not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or np.any(probabilities > 1)
            or not np.isclose(
                probabilities.sum(), 1.0, atol=1e-3
            )
        ):
            raise ValueError("Invalid class probabilities.")

        grade = int(np.argmax(probabilities))

        return {
            "image_name": Path(image_path).name,
            "severity_grade": grade,
            "label": CLASS_NAMES[grade],
            "raw_confidence": float(probabilities[grade]),
            "probabilities": probabilities.tolist(),
            "calibrated_confidence": None,
            "confidence_flag": "not_calibrated",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        required=True,
        help="Path to a fundus image",
    )
    args = parser.parse_args()

    adapter = DRModelAdapter()
    result = adapter.predict(args.image)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()