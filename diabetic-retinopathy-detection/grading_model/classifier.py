from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


ICDR_LABELS = (
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR",
)
DEFAULT_TEMPERATURE = 1.0


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


@dataclass
class TemperatureScaler:
    temperature: float = DEFAULT_TEMPERATURE

    def __post_init__(self) -> None:
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be a finite positive number")

    def probabilities(self, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64)
        if logits.ndim == 1:
            logits = logits[None, :]
        return _softmax(logits / self.temperature)

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if logits.ndim != 2 or logits.shape[0] != labels.shape[0]:
            raise ValueError("logits must be [samples, classes] and match labels")
        if np.any(labels < 0) or np.any(labels >= logits.shape[1]):
            raise ValueError("labels must be valid class indices")

        def loss(log_temperature: float) -> float:
            probabilities = _softmax(logits / np.exp(log_temperature))
            selected = probabilities[np.arange(len(labels)), labels]
            return float(-np.mean(np.log(np.clip(selected, 1e-12, 1.0))))

        grid = np.linspace(np.log(0.05), np.log(10.0), 161)
        losses = np.asarray([loss(value) for value in grid])
        self.temperature = float(np.exp(grid[int(np.argmin(losses))]))
        return self.temperature

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"temperature": self.temperature}, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "TemperatureScaler":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(float(payload["temperature"]))


class LesionAwareSeverityClassifier:
    def __init__(
        self,
        model: Any,
        temperature: float = DEFAULT_TEMPERATURE,
        low_confidence_threshold: float = 0.60,
        lesion_predictor: Callable[[Any], Mapping[str, float]] | None = None,
    ) -> None:
        self.model = model
        self.temperature_scaler = TemperatureScaler(temperature)
        if not 0.0 <= low_confidence_threshold <= 1.0:
            raise ValueError("low_confidence_threshold must be between 0 and 1")
        self.low_confidence_threshold = low_confidence_threshold
        self.lesion_predictor = lesion_predictor

    def _lesion_adjustment(
    self,
    lesion_evidence: Mapping[str, float] | None
    ) -> np.ndarray:
        """Return neutral adjustment until a trained lesion model is connected."""
        return np.zeros(5, dtype=np.float64)
        
        

    def predict(
        self,
        image: Any,
        enhanced_image: Any | None = None,
        lesion_evidence: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        model_input = enhanced_image if enhanced_image is not None else image
        if lesion_evidence is None and self.lesion_predictor is not None:
            lesion_evidence = self.lesion_predictor(model_input)

        probabilities = np.asarray(self.model.predict(model_input, verbose=0))
        if probabilities.ndim != 2 or probabilities.shape[1] != 5:
            raise ValueError("severity model must return probabilities for five ICDR classes")
        probabilities = np.clip(probabilities[0].astype(np.float64), 1e-8, 1.0)
        probabilities /= probabilities.sum()
        raw_logits = np.log(probabilities) + self._lesion_adjustment(lesion_evidence)
        raw_probabilities = _softmax(raw_logits[None, :])[0]
        calibrated_probabilities = self.temperature_scaler.probabilities(raw_logits)[0]
        severity_grade = int(np.argmax(calibrated_probabilities))
        calibrated_confidence = float(calibrated_probabilities[severity_grade])

        return {
            "severity_grade": severity_grade,
            "raw_confidence": float(np.max(raw_probabilities)),
            "calibrated_confidence": calibrated_confidence,
            "confidence_flag": (
                
                "low" if calibrated_confidence < self.low_confidence_threshold - 1e-9 else "ok"
            ),
            "probabilities": calibrated_probabilities.tolist(),
            "raw_probabilities": raw_probabilities.tolist(),
            "lesion_evidence": dict(lesion_evidence or {}),
        }