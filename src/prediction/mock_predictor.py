"""Development-only mock predictor used before a real model is available."""

from __future__ import annotations

from src.core.contracts import GenericFeatureRecord, PredictionResult
from src.prediction.predictor import Predictor


class MockPredictor(Predictor):
    """Simple deterministic placeholder predictor.

    This class exists to allow the rest of the system to be built and tested
    before a trained machine learning model is available. It intentionally does
    not pretend to reproduce N-BaIoT behavior or any final feature schema.
    """

    model_name: str = "mock-model"
    source: str = "mock"

    def predict(self, feature_record: GenericFeatureRecord) -> PredictionResult:
        """Return a deterministic development-only prediction."""
        if not isinstance(feature_record, GenericFeatureRecord):
            raise TypeError("feature_record must be a GenericFeatureRecord instance")

        numeric_values = [
            float(value)
            for value in feature_record.features.values()
            if isinstance(value, (int, float, bool))
        ]

        total_signal = sum(abs(value) for value in numeric_values)

        if total_signal > 0:
            prediction = "attack"
            probability = min(0.99, 0.55 + (total_signal / max(1.0, total_signal + 100.0)))
            notes = "Development-only mock prediction; not a trained model."
        else:
            prediction = "normal"
            probability = 0.05
            notes = "Development-only mock prediction; not a trained model."

        return PredictionResult(
            flow_id=str(feature_record.metadata.get("flow_id", "mock-flow")),
            prediction=prediction,
            probability=probability,
            timestamp=float(feature_record.metadata.get("timestamp", 0.0)),
            model_name=self.model_name,
            source=self.source,
            notes=notes,
        )
