"""Random Forest predictor integration for the project.

This implementation follows the existing Predictor contract but remains
independent from the pipeline and storage layers. It validates model artifact
presence and feature compatibility before inference. The real teammate artifact
is still the source of truth for the final model schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import joblib

from src.core.contracts import GenericFeatureRecord, PredictionResult
from src.prediction.predictor import Predictor


class ModelCompatibilityError(ValueError):
    """Raised when the model artifact or feature schema is incompatible."""


class RandomForestPredictor(Predictor):
    """Wrap a trained Random Forest model behind the Predictor interface."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        model: Any | None = None,
        feature_names: Sequence[str] | None = None,
        label_mapping: dict[Any, str] | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else None
        self.model = model
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.label_mapping = label_mapping or {}

    def load_model(self) -> Any:
        """Load the trained model artifact or use the provided in-memory model."""
        if self.model is not None:
            return self.model

        if self.model_path is None:
            raise ValueError("A model path must be provided to load a Random Forest model.")

        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")

        loaded_model = joblib.load(path)
        if not hasattr(loaded_model, "predict"):
            raise ValueError("Loaded model does not expose a predict method.")

        self.model = loaded_model
        return self.model

    def _resolve_feature_names(self, model: Any) -> list[str] | None:
        """Determine the model's expected feature names, when available."""
        feature_names = getattr(model, "feature_names_in_", None)
        if feature_names is not None:
            return [str(name) for name in feature_names]
        return list(self.feature_names) if self.feature_names is not None else None

    def _coerce_numeric(self, value: Any) -> float:
        """Convert supported primitive types to float without silently accepting arbitrary objects."""
        if isinstance(value, bool):
            return float(int(value))
        if isinstance(value, (int, float)):
            return float(value)
        raise TypeError(f"Feature value {value!r} is not numeric.")

    def _feature_vector_for_model(self, feature_record: GenericFeatureRecord) -> list[float]:
        """Convert GenericFeatureRecord into the model's expected numeric vector."""
        if not isinstance(feature_record, GenericFeatureRecord):
            raise TypeError("feature_record must be a GenericFeatureRecord instance")

        features = feature_record.features
        if not isinstance(features, dict):
            raise TypeError("feature_record.features must be a dictionary")

        model = self.load_model()
        expected_names = self._resolve_feature_names(model)

        if expected_names is not None:
            ordered: list[float] = []
            missing = [name for name in expected_names if name not in features]
            if missing:
                raise ModelCompatibilityError(
                    "Model schema requires features missing from GenericFeatureRecord: "
                    + ", ".join(missing)
                )
            for name in expected_names:
                ordered.append(self._coerce_numeric(features[name]))
            return ordered

        if self.feature_names is not None:
            ordered = []
            for name in self.feature_names:
                if name not in features:
                    raise ModelCompatibilityError(f"Required feature '{name}' is missing.")
                ordered.append(self._coerce_numeric(features[name]))
            return ordered

        if not features:
            raise ModelCompatibilityError("GenericFeatureRecord.features is empty.")

        return [self._coerce_numeric(value) for value in features.values()]

    def _resolve_prediction_label(self, raw_prediction: Any, model: Any) -> str:
        """Convert a model prediction into a string label using the model's classes or mapping."""
        if self.label_mapping:
            for key, label in self.label_mapping.items():
                if raw_prediction == key:
                    return str(label)

        classes = getattr(model, "classes_", None)
        if classes is not None:
            for label in classes:
                if raw_prediction == label:
                    return str(label)

        return str(raw_prediction)

    def predict(self, feature_record: GenericFeatureRecord) -> PredictionResult:
        """Predict using the loaded Random Forest model and return a standard result."""
        model = self.load_model()
        feature_vector = self._feature_vector_for_model(feature_record)

        prediction_value = model.predict([feature_vector])[0]
        label = self._resolve_prediction_label(prediction_value, model)

        probability = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([feature_vector])[0]
            classes = list(getattr(model, "classes_", []))
            if classes:
                for candidate, value in zip(classes, probabilities):
                    if candidate == prediction_value:
                        probability = float(value)
                        break

        return PredictionResult(
            flow_id=str(feature_record.metadata.get("flow_id", "unknown-flow")),
            prediction=label,
            probability=probability,
            timestamp=float(feature_record.metadata.get("timestamp", 0.0)),
            model_name=type(model).__name__,
            source="random_forest",
            notes="Model prediction generated by RandomForestPredictor.",
        )
