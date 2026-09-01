"""Random Forest predictor integration for the project.

This implementation follows the existing Predictor contract and provides
production Random Forest inference using the project's trained model.

The predictor:
    - Loads the configured Random Forest model.
    - Validates model feature compatibility.
    - Preserves the exact feature order expected by the model.
    - Passes named features to scikit-learn using a pandas DataFrame.
    - Converts raw model classes into project-level labels.
    - Returns a standard PredictionResult.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import joblib
import pandas as pd

from src.core.contracts import GenericFeatureRecord, PredictionResult
from src.prediction.predictor import Predictor


class ModelCompatibilityError(ValueError):
    """Raised when the model artifact or feature schema is incompatible."""


class RandomForestPredictor(Predictor):
    """Wrap a trained Random Forest model behind the Predictor interface."""

    DEFAULT_MODEL_PATH = (
        Path(__file__).resolve().parents[2]
        / "ml"
        / "model"
        / "random_forest.joblib"
    )

    DEFAULT_LABEL_MAPPING = {
        0: "Normal",
        1: "Attack",
    }

    def __init__(
        self,
        model_path: str | Path | None = None,
        model: Any | None = None,
        feature_names: Sequence[str] | None = None,
        label_mapping: dict[Any, str] | None = None,
    ) -> None:
        self.model_path = (
            Path(model_path)
            if model_path is not None
            else self.DEFAULT_MODEL_PATH
        )

        self.model = model

        self.feature_names = (
            list(feature_names)
            if feature_names is not None
            else None
        )

        self.label_mapping = (
            dict(label_mapping)
            if label_mapping is not None
            else dict(self.DEFAULT_LABEL_MAPPING)
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> Any:
        """Load the trained model artifact or use the provided model."""

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
        """Determine the exact feature names expected by the model."""

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
    # Numeric validation
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_numeric(value: Any) -> float:
        """Convert supported primitive values to float."""

        if isinstance(value, bool):
            return float(int(value))

        if isinstance(value, (int, float)):
            return float(value)

        raise TypeError(
            f"Feature value {value!r} is not numeric."
        )

    # ------------------------------------------------------------------
    # Feature preparation
    # ------------------------------------------------------------------

    def _feature_vector_for_model(
        self,
        feature_record: GenericFeatureRecord,
    ) -> tuple[list[str], list[float]]:
        """Build the exact named feature vector expected by the model."""

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
        # Model exposes feature_names_in_
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

            ordered_values = [
                self._coerce_numeric(features[name])
                for name in expected_names
            ]

            return expected_names, ordered_values

        # --------------------------------------------------------------
        # Explicit feature schema supplied to predictor
        # --------------------------------------------------------------

        if self.feature_names is not None:
            expected_names = list(self.feature_names)

            missing = [
                name
                for name in expected_names
                if name not in features
            ]

            if missing:
                raise ModelCompatibilityError(
                    "Required features are missing: "
                    + ", ".join(missing)
                )

            ordered_values = [
                self._coerce_numeric(features[name])
                for name in expected_names
            ]

            return expected_names, ordered_values

        # --------------------------------------------------------------
        # Last-resort dictionary order
        # --------------------------------------------------------------

        names = list(features.keys())

        values = [
            self._coerce_numeric(value)
            for value in features.values()
        ]

        return names, values

    # ------------------------------------------------------------------
    # Prediction label
    # ------------------------------------------------------------------

    def _resolve_prediction_label(
        self,
        raw_prediction: Any,
        model: Any,
    ) -> str:
        """Convert a raw model class into a project-level label."""

        # Explicit/project label mapping takes priority.
        for key, label in self.label_mapping.items():
            try:
                if raw_prediction == key:
                    return str(label)
            except Exception:
                continue

        # If there is no mapping, fall back to the model's class name.
        classes = getattr(
            model,
            "classes_",
            None,
        )

        if classes is not None:
            for class_value in classes:
                try:
                    if raw_prediction == class_value:
                        return str(class_value)
                except Exception:
                    continue

        return str(raw_prediction)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        feature_record: GenericFeatureRecord,
    ) -> PredictionResult:
        """Predict using the Random Forest model."""

        model = self.load_model()

        feature_names, feature_vector = (
            self._feature_vector_for_model(
                feature_record
            )
        )

        # --------------------------------------------------------------
        # IMPORTANT:
        #
        # The trained Random Forest contains feature_names_in_.
        #
        # Therefore we pass a DataFrame with the exact same names and
        # order. This prevents sklearn's:
        #
        # "X does not have valid feature names"
        #
        # warning.
        # --------------------------------------------------------------

        X = pd.DataFrame(
            [feature_vector],
            columns=feature_names,
        )

        # --------------------------------------------------------------
        # Prediction
        # --------------------------------------------------------------

        prediction_value = model.predict(X)[0]

        label = self._resolve_prediction_label(
            prediction_value,
            model,
        )

        # --------------------------------------------------------------
        # Probability
        # --------------------------------------------------------------

        probability: float | None = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[0]

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
                try:
                    if candidate == prediction_value:
                        probability = float(value)
                        break
                except Exception:
                    continue

        # --------------------------------------------------------------
        # Metadata
        # --------------------------------------------------------------

        flow_id = str(
            feature_record.metadata.get(
                "flow_id",
                "unknown-flow",
            )
        )

        timestamp = float(
            feature_record.metadata.get(
                "timestamp",
                0.0,
            )
        )

        return PredictionResult(
            flow_id=flow_id,
            prediction=label,
            probability=probability,
            timestamp=timestamp,
            model_name=type(model).__name__,
            source="random_forest",
            notes=(
                "Model prediction generated by "
                "RandomForestPredictor."
            ),
        )