"""Prediction interfaces and implementations."""

from .mock_predictor import MockPredictor
from .predictor import Predictor

__all__ = ["Predictor", "MockPredictor"]
