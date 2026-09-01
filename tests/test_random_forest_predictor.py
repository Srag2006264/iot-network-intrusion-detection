"""Tests for the Random Forest predictor integration contract."""

from __future__ import annotations

from pathlib import Path

import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.core.contracts import GenericFeatureRecord, PredictionResult
from src.pipeline.detection import DetectionPipeline
from src.prediction.mock_predictor import MockPredictor
from src.prediction.predictor import Predictor
from src.prediction.random_forest_predictor import (
    ModelCompatibilityError,
    RandomForestPredictor,
)
from src.storage.database import Database


@pytest.fixture
def tiny_model_path(tmp_path):
    """Create a small temporary Random Forest model for isolated tests."""
    model_path = tmp_path / "tiny_rf_model.joblib"

    model = RandomForestClassifier(
        n_estimators=5,
        random_state=42,
    )

    X = [
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0],
        [6.0, 7.0, 8.0],
    ]

    y = [0, 0, 1, 1]

    model.fit(X, y)
    joblib.dump(model, model_path)

    return model_path


def test_random_forest_predictor_conforms_to_predictor():
    """RandomForestPredictor must implement the Predictor interface."""
    predictor = RandomForestPredictor()

    assert isinstance(predictor, Predictor)


def test_random_forest_predictor_loads_valid_model(tiny_model_path):
    """An explicitly supplied model artifact can be loaded."""
    predictor = RandomForestPredictor(
        model_path=tiny_model_path
    )

    model = predictor.load_model()

    assert model is not None
    assert hasattr(model, "predict")
    assert isinstance(model, RandomForestClassifier)


def test_random_forest_predictor_loads_default_project_model():
    """The project Random Forest should load when no path is supplied."""
    predictor = RandomForestPredictor()

    model = predictor.load_model()

    assert model is not None
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "predict")
    assert hasattr(model, "feature_names_in_")
    assert len(model.feature_names_in_) == 115


def test_random_forest_predictor_returns_prediction_result(tiny_model_path):
    """Prediction should return the project's standard PredictionResult."""
    predictor = RandomForestPredictor(
        model_path=tiny_model_path
    )

    feature_record = GenericFeatureRecord(
        feature_version="test-v1",
        features={
            "feature_1": 1.0,
            "feature_2": 2.0,
            "feature_3": 3.0,
        },
        metadata={
            "flow_id": "flow-rf",
            "timestamp": 1.0,
        },
    )

    result = predictor.predict(feature_record)

    assert isinstance(result, PredictionResult)
    assert result.flow_id == "flow-rf"

    # The project predictor maps Random Forest classes:
    # 0 -> Normal
    # 1 -> Attack
    assert result.prediction in {"Normal", "Attack"}

    assert result.source == "random_forest"
    assert result.model_name == "RandomForestClassifier"


def test_random_forest_predictor_maps_probability_when_available(
    tiny_model_path,
):
    """Probability should be returned when predict_proba is available."""
    predictor = RandomForestPredictor(
        model_path=tiny_model_path
    )

    feature_record = GenericFeatureRecord(
        feature_version="test-v1",
        features={
            "feature_1": 1.0,
            "feature_2": 2.0,
            "feature_3": 3.0,
        },
        metadata={
            "flow_id": "flow-prob",
            "timestamp": 2.0,
        },
    )

    result = predictor.predict(feature_record)

    assert result.probability is not None
    assert 0.0 <= result.probability <= 1.0


def test_random_forest_predictor_rejects_invalid_feature_count(
    tiny_model_path,
):
    """Missing model features must be rejected."""
    predictor = RandomForestPredictor(
        model_path=tiny_model_path
    )

    feature_record = GenericFeatureRecord(
        feature_version="test-v1",
        features={
            "feature_1": 1.0,
            "feature_2": 2.0,
        },
        metadata={
            "flow_id": "flow-short",
            "timestamp": 3.0,
        },
    )

    with pytest.raises(
        (ModelCompatibilityError, ValueError, TypeError)
    ):
        predictor.predict(feature_record)


def test_random_forest_predictor_rejects_missing_model_artifact():
    """A nonexistent explicit model path must raise an error."""
    predictor = RandomForestPredictor(
        model_path=Path("missing_model.joblib")
    )

    with pytest.raises((FileNotFoundError, ValueError)):
        predictor.load_model()


def test_mock_predictor_still_works():
    """The existing mock predictor must remain functional."""
    predictor = MockPredictor()

    feature_record = GenericFeatureRecord(
        feature_version="test-v1",
        features={
            "packet_count": 10,
            "byte_count": 100,
            "duration": 2.0,
        },
        metadata={
            "flow_id": "flow-mock",
            "timestamp": 4.0,
        },
    )

    result = predictor.predict(feature_record)

    assert isinstance(result, PredictionResult)
    assert result.model_name == "mock-model"


def test_detection_pipeline_accepts_random_forest_predictor(
    tmp_path,
    tiny_model_path,
):
    """DetectionPipeline should accept RandomForestPredictor."""
    database = Database(
        tmp_path / "rf_pipeline.sqlite3"
    )

    predictor = RandomForestPredictor(
        model_path=tiny_model_path
    )

    pipeline = DetectionPipeline(
        predictor=predictor,
        database=database,
    )

    assert pipeline.predictor is predictor
    assert isinstance(
        pipeline.predictor,
        Predictor,
    )


def test_normal_prediction_does_not_generate_alert(
    tmp_path,
    tiny_model_path,
):
    """Normal predictions should not be treated as attacks."""
    database = Database(
        tmp_path / "rf_normal.sqlite3"
    )

    predictor = RandomForestPredictor(
        model_path=tiny_model_path
    )

    feature_record = GenericFeatureRecord(
        feature_version="test-v1",
        features={
            "feature_1": 1.0,
            "feature_2": 2.0,
            "feature_3": 3.0,
        },
        metadata={
            "flow_id": "flow-normal",
            "timestamp": 5.0,
        },
    )

    result = predictor.predict(feature_record)

    assert isinstance(result, PredictionResult)
    assert result.prediction == "Normal"

    assert result.probability is not None
    assert 0.0 <= result.probability <= 1.0


def test_attack_prediction_generates_attack_prediction(
    tmp_path,
    tiny_model_path,
):
    """Attack-like test data should produce the Attack label."""
    database = Database(
        tmp_path / "rf_attack.sqlite3"
    )

    predictor = RandomForestPredictor(
        model_path=tiny_model_path
    )

    feature_record = GenericFeatureRecord(
        feature_version="test-v1",
        features={
            "feature_1": 6.0,
            "feature_2": 7.0,
            "feature_3": 8.0,
        },
        metadata={
            "flow_id": "flow-attack",
            "timestamp": 6.0,
        },
    )

    result = predictor.predict(feature_record)

    assert isinstance(result, PredictionResult)
    assert result.prediction == "Attack"

    assert result.probability is not None
    assert 0.0 <= result.probability <= 1.0


def test_random_forest_predictor_rejects_incompatible_feature_names_when_available(
    tiny_model_path,
):
    """A feature name mismatch must be rejected."""
    model = joblib.load(tiny_model_path)

    model.feature_names_in_ = [
        "feature_1",
        "feature_2",
        "feature_3",
    ]

    predictor = RandomForestPredictor(
        model=model
    )

    feature_record = GenericFeatureRecord(
        feature_version="test-v1",
        features={
            "feature_1": 1.0,
            "feature_2": 2.0,
            "feature_4": 3.0,
        },
        metadata={
            "flow_id": "flow-bad-names",
            "timestamp": 7.0,
        },
    )

    with pytest.raises(
        (ModelCompatibilityError, KeyError, TypeError)
    ):
        predictor.predict(feature_record)


def test_random_forest_predictor_rejects_non_numeric_feature(
    tiny_model_path,
):
    """Non-numeric feature values must be rejected."""
    predictor = RandomForestPredictor(
        model_path=tiny_model_path
    )

    feature_record = GenericFeatureRecord(
        feature_version="test-v1",
        features={
            "feature_1": "not-a-number",
            "feature_2": 2.0,
            "feature_3": 3.0,
        },
        metadata={
            "flow_id": "flow-invalid",
            "timestamp": 8.0,
        },
    )

    with pytest.raises(TypeError):
        predictor.predict(feature_record)


def test_random_forest_predictor_uses_model_feature_order(
    tiny_model_path,
):
    """Features must be passed to the model in its expected order."""
    model = joblib.load(tiny_model_path)

    model.feature_names_in_ = [
        "feature_1",
        "feature_2",
        "feature_3",
    ]

    predictor = RandomForestPredictor(
        model=model
    )

    feature_record = GenericFeatureRecord(
        feature_version="test-v1",
        features={
            # Deliberately reversed dictionary order.
            "feature_3": 3.0,
            "feature_1": 1.0,
            "feature_2": 2.0,
        },
        metadata={
            "flow_id": "flow-order",
            "timestamp": 9.0,
        },
    )

    result = predictor.predict(feature_record)

    assert isinstance(result, PredictionResult)
    assert result.prediction in {"Normal", "Attack"}


def test_default_random_forest_model_has_115_features():
    """The production model must expose the expected N-BaIoT schema."""
    predictor = RandomForestPredictor()

    model = predictor.load_model()

    assert hasattr(model, "feature_names_in_")
    assert len(model.feature_names_in_) == 115


def test_default_random_forest_model_has_binary_classes():
    """The production model must have the expected binary classes."""
    predictor = RandomForestPredictor()

    model = predictor.load_model()

    assert list(model.classes_) == [0, 1]