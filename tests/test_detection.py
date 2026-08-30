"""Tests for the detection pipeline."""

from __future__ import annotations

from pathlib import Path

from src.core.contracts import AlertRecord, FlowRecord, GenericFeatureRecord, PredictionResult
from src.network.features import FlowFeatureExtractor
from src.pipeline.detection import DetectionPipeline
from src.prediction.mock_predictor import MockPredictor
from src.prediction.predictor import Predictor
from src.storage.database import Database


def _make_flow(flow_id: str, packet_count: int = 10, byte_count: int = 500, duration: float = 2.0) -> FlowRecord:
    return FlowRecord(
        flow_id=flow_id,
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        src_port=5000,
        dst_port=80,
        protocol="tcp",
        start_time=1.0,
        end_time=1.0 + duration,
        duration=duration,
        packet_count=packet_count,
        byte_count=byte_count,
        summary_stats={"min_packet_size": 20, "max_packet_size": 80},
    )


def test_flow_can_pass_through_feature_extraction_and_prediction(tmp_path):
    db_path = tmp_path / "pipeline.sqlite3"
    database = Database(db_path)
    pipeline = DetectionPipeline(
        feature_extractor=FlowFeatureExtractor(),
        predictor=MockPredictor(),
        database=database,
    )

    flow = _make_flow("flow-01")
    result = pipeline.process_flow(flow)

    assert result["persisted"] is True
    assert isinstance(result["feature_record"], GenericFeatureRecord)
    assert isinstance(result["prediction"], PredictionResult)


def test_mock_predictor_can_be_injected_into_pipeline(tmp_path):
    db_path = tmp_path / "pipeline.sqlite3"
    database = Database(db_path)
    predictor = MockPredictor()
    pipeline = DetectionPipeline(predictor=predictor, database=database)

    assert pipeline.predictor is predictor
    assert isinstance(pipeline.predictor, Predictor)


def test_prediction_result_is_produced_correctly(tmp_path):
    db_path = tmp_path / "pipeline.sqlite3"
    database = Database(db_path)
    pipeline = DetectionPipeline(predictor=MockPredictor(), database=database)

    flow = _make_flow("flow-02", packet_count=20, byte_count=1000, duration=4.0)
    result = pipeline.process_flow(flow)

    prediction = result["prediction"]
    assert prediction.flow_id == "flow-02"
    assert prediction.prediction in {"normal", "attack"}
    assert prediction.probability is not None


def test_normal_prediction_does_not_create_an_alert(tmp_path):
    db_path = tmp_path / "pipeline.sqlite3"
    database = Database(db_path)
    pipeline = DetectionPipeline(predictor=MockPredictor(), database=database)

    flow = _make_flow("flow-03", packet_count=0, byte_count=0, duration=0.0)
    result = pipeline.process_flow(flow)

    assert result["alert"] is None
    assert len(database.get_all_alerts()) == 0


def test_attack_prediction_creates_alert_record(tmp_path):
    db_path = tmp_path / "pipeline.sqlite3"
    database = Database(db_path)
    pipeline = DetectionPipeline(predictor=MockPredictor(), database=database)

    flow = _make_flow("flow-04", packet_count=100, byte_count=5000, duration=1.0)
    result = pipeline.process_flow(flow)

    assert result["alert"] is not None
    assert isinstance(result["alert"], AlertRecord)
    assert result["alert"].prediction in {"normal", "attack"}


def test_flow_data_is_persisted(tmp_path):
    db_path = tmp_path / "pipeline.sqlite3"
    database = Database(db_path)
    pipeline = DetectionPipeline(predictor=MockPredictor(), database=database)

    flow = _make_flow("flow-05")
    pipeline.process_flow(flow)

    stored = database.get_flow("flow-05")
    assert stored is not None
    assert stored.flow_id == "flow-05"


def test_prediction_data_is_persisted(tmp_path):
    db_path = tmp_path / "pipeline.sqlite3"
    database = Database(db_path)
    pipeline = DetectionPipeline(predictor=MockPredictor(), database=database)

    flow = _make_flow("flow-06")
    pipeline.process_flow(flow)

    prediction = database.get_prediction("flow-06")
    assert prediction is not None
    assert prediction.flow_id == "flow-06"


def test_alert_data_is_persisted_when_attack_is_predicted(tmp_path):
    db_path = tmp_path / "pipeline.sqlite3"
    database = Database(db_path)
    pipeline = DetectionPipeline(predictor=MockPredictor(), database=database)

    flow = _make_flow("flow-07", packet_count=200, byte_count=9000, duration=1.0)
    pipeline.process_flow(flow)

    alerts = database.get_all_alerts()
    assert len(alerts) >= 1


def test_normal_predictions_do_not_create_false_alerts(tmp_path):
    db_path = tmp_path / "pipeline.sqlite3"
    database = Database(db_path)
    pipeline = DetectionPipeline(predictor=MockPredictor(), database=database)

    flow = _make_flow("flow-08", packet_count=0, byte_count=0, duration=0.0)
    pipeline.process_flow(flow)

    assert len(database.get_all_alerts()) == 0


def test_pipeline_dependencies_are_injectable():
    extractor = FlowFeatureExtractor()
    predictor = MockPredictor()
    database = Database(Path("/tmp/test_pipeline.sqlite3"))
    pipeline = DetectionPipeline(feature_extractor=extractor, predictor=predictor, database=database)

    assert pipeline.feature_extractor is extractor
    assert pipeline.predictor is predictor
    assert pipeline.database is database


def test_pipeline_does_not_directly_depend_on_mock_predictor_class():
    pipeline = DetectionPipeline(predictor=MockPredictor())
    assert pipeline.predictor is not None
    assert not isinstance(pipeline.predictor, type)


def test_pipeline_processes_multiple_flows_independently(tmp_path):
    db_path = tmp_path / "pipeline.sqlite3"
    database = Database(db_path)
    pipeline = DetectionPipeline(predictor=MockPredictor(), database=database)

    first = pipeline.process_flow(_make_flow("flow-09"))
    second = pipeline.process_flow(_make_flow("flow-10", packet_count=50, byte_count=2500, duration=1.5))

    assert first["prediction"].flow_id == "flow-09"
    assert second["prediction"].flow_id == "flow-10"
    assert len(database.get_all_predictions()) == 2
