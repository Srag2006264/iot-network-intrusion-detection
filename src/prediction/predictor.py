"""Abstraction for prediction components in the IDS pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.contracts import GenericFeatureRecord, PredictionResult


class Predictor(ABC):
    """Abstract interface for inference implementations.

    The interface is intentionally model-agnostic. Any concrete predictor,
    whether a mock predictor or a future Random Forest predictor, must return a
    standard PredictionResult for a given GenericFeatureRecord.
    """

    @abstractmethod
    def predict(self, feature_record: GenericFeatureRecord) -> PredictionResult:
        """Return a prediction result for the supplied feature record."""

