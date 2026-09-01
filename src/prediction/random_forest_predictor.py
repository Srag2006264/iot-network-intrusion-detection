"""Random Forest predictor integration for the project.

This module wraps the trained Random Forest model behind the project's
model-agnostic Predictor interface.

The predictor:
    1. Loads the trained Random Forest artifact.
    2. Validates the expected feature schema.
    3. Converts GenericFeatureRecord into the correct feature order.
    4. Performs prediction and probability estimation.
    5. Returns the project's standard PredictionResult.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import joblib

from src.core.contracts import GenericFeatureRecord, PredictionResult
from src.prediction.predictor import Predictor


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "model"
    / "random_forest.joblib"
)


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
        # If no model path is supplied, use the project's trained
        # Random Forest model automatically.
        self.model_path = (
            Path(model_path)
            if model_path is not None
            else DEFAULT_MODEL_PATH
        )

        self.model = model
        self.feature_names = (
            list(feature_names)
            if feature_names is not None
            else None
        )

        # The trained model uses:
        #   0 = Normal
        #   1 = Attack
        #
        # Keep this mapping explicit so PredictionResult contains
        # human-readable labels.
        self.label_mapping = (
            label_mapping
            if label_mapping is not None
            else {
                0: "Normal",
                1: "Attack",
            }
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> Any:
        """Load the trained Random Forest model."""

        if self.model is not None:
            return self.model

        if self.model_path is None:
            raise ValueError(
                "A model path or model instance must be provided."
            )

        path = Path(self.model_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {path}"
            )

        loaded_model = joblib.load(path)

        if not hasattr(loaded_model, "predict"):
            raise ValueError(
                "Loaded model does not expose a predict method."
            )

        self.model = loaded_model

        return self.model

    # ------------------------------------------------------------------
    # Feature schema
    # ------------------------------------------------------------------

    def _resolve_feature_names(
        self,
        model: Any,
    ) -> list[str] | None:
        """Determine the model's expected feature names."""

        model_feature_names = getattr(
            model,
            "feature_names_in_",
            None,
        )

        if model_feature_names is not None:
            return [
                str(name)
                for name in model_feature_names
            ]

        if self.feature_names is not None:
            return list(self.feature_names)

        return None

    # ------------------------------------------------------------------
    # Feature validation
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_numeric(value: Any) -> float:
        """Convert supported numeric values to float."""

        if isinstance(value, bool):
            return float(int(value))

        if isinstance(value, (int, float)):
            return float(value)

        raise TypeError(
            f"Feature value {value!r} is not numeric."
        )

    def _feature_vector_for_model(
        self,
        feature_record: GenericFeatureRecord,
    ) -> list[float]:
        """Convert GenericFeatureRecord into the model's expected vector."""

        if not isinstance(
            feature_record,
            GenericFeatureRecord,
        ):
            raise TypeError(
                "feature_record must be a GenericFeatureRecord instance"
            )

        features = feature_record.features

        if not isinstance(features, dict):
            raise TypeError(
                "feature_record.features must be a dictionary"
            )

        if not features:
            raise ModelCompatibilityError(
                "GenericFeatureRecord.features is empty."
            )

        model = self.load_model()

        expected_names = self._resolve_feature_names(model)

        # --------------------------------------------------------------
        # Model provides feature_names_in_
        # --------------------------------------------------------------

        if expected_names is not None:
            missing = [
                name
                for name in expected_names
                if name not in features
            ]

            if missing:
                raise ModelCompatibilityError(
                    "Model schema requires features missing from "
                    "GenericFeatureRecord: "
                    + ", ".join(missing)
                )

            ordered: list[float] = []

            for name in expected_names:
                ordered.append(
                    self._coerce_numeric(
                        features[name]
                    )
                )

            return ordered

        # --------------------------------------------------------------
        # Explicit feature_names supplied
        # --------------------------------------------------------------

        if self.feature_names is not None:
            ordered = []

            for name in self.feature_names:
                if name not in features:
                    raise ModelCompatibilityError(
                        f"Required feature '{name}' is missing."
                    )

                ordered.append(
                    self._coerce_numeric(
                        features[name]
                    )
                )

            return ordered

        # --------------------------------------------------------------
        # Fallback
        # --------------------------------------------------------------

        return [
            self._coerce_numeric(value)
            for value in features.values()
        ]

    # ------------------------------------------------------------------
    # Prediction label
    # ------------------------------------------------------------------

    def _resolve_prediction_label(
        self,
        raw_prediction: Any,
        model: Any,
    ) -> str:
        """Convert raw model output into a human-readable label."""

        # Explicit project mapping takes priority.
        if self.label_mapping:
            for key, label in self.label_mapping.items():
                if raw_prediction == key:
                    return str(label)

        # Fall back to model classes.
        classes = getattr(
            model,
            "classes_",
            None,
        )

        if classes is not None:
            for label in classes:
                if raw_prediction == label:
                    return str(label)

        return str(raw_prediction)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        feature_record: GenericFeatureRecord,
    ) -> PredictionResult:
        """Run Random Forest inference and return PredictionResult."""

        model = self.load_model()

        feature_vector = (
            self._feature_vector_for_model(
                feature_record
            )
        )

        # --------------------------------------------------------------
        # Prediction
        # --------------------------------------------------------------

        prediction_value = model.predict(
            [feature_vector]
        )[0]

        prediction_label = (
            self._resolve_prediction_label(
                prediction_value,
                model,
            )
        )

        # --------------------------------------------------------------
        # Probability
        # --------------------------------------------------------------

        probability: float | None = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(
                [feature_vector]
            )[0]

            classes = list(
                getattr(
                    model,
                    "classes_",
                    [],
                )
            )

            for candidate, value in zip(
                classes,
                probabilities,
            ):
                if candidate == prediction_value:
                    probability = float(value)
                    break

        # --------------------------------------------------------------
        # PredictionResult
        # --------------------------------------------------------------

        return PredictionResult(
            flow_id=str(
                feature_record.metadata.get(
                    "flow_id",
                    "unknown-flow",
                )
            ),
            prediction=prediction_label,
            probability=probability,
            timestamp=float(
                feature_record.metadata.get(
                    "timestamp",
                    0.0,
                )
            ),
            model_name=type(model).__name__,
            source="random_forest",
            notes=(
                "Model prediction generated by "
                "RandomForestPredictor."
            ),
        )