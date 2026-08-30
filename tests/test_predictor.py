"""Tests for predictor abstractions and mock predictor behavior."""

from src.core.contracts import GenericFeatureRecord, PredictionResult
from src.prediction.mock_predictor import MockPredictor
from src.prediction.predictor import Predictor


def test_mock_predictor_satisfies_predictor_contract():
    predictor = MockPredictor()

    assert isinstance(predictor, Predictor)


def test_mock_predictor_returns_prediction_result():
    predictor = MockPredictor()
    feature_record = GenericFeatureRecord(
        feature_version="1.0",
        features={"packet_count": 12, "byte_count": 1024, "duration": 2.5},
        metadata={"flow_id": "flow-123", "timestamp": 100.0},
    )

    result = predictor.predict(feature_record)

    assert isinstance(result, PredictionResult)
    assert result.flow_id == "flow-123"
    assert result.model_name == "mock-model"
    assert result.source == "mock"
    assert result.prediction in {"normal", "attack"}
    assert result.probability is not None
    assert 0.0 <= result.probability <= 1.0


def test_mock_predictor_is_deterministic_for_same_input():
    predictor = MockPredictor()
    feature_record = GenericFeatureRecord(
        feature_version="1.0",
        features={"packet_count": 8, "byte_count": 400, "duration": 1.0},
        metadata={"flow_id": "flow-456", "timestamp": 50.0},
    )

    first = predictor.predict(feature_record)
    second = predictor.predict(feature_record)

    assert first == second
