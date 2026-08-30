"""Stage 8A: end-to-end integration validation with the mock predictor.

This test verifies that a flow moves through the real application interfaces
without requiring the teammate's Random Forest model or any live network
traffic. The deterministic data path is:

FlowRecord -> Feature extraction -> GenericFeatureRecord -> MockPredictor
-> PredictionResult -> DetectionPipeline -> SQLite -> SecurityDashboard
"""

from __future__ import annotations

from src.core.contracts import FlowRecord
from src.dashboard.app import SecurityDashboard
from src.network.features import FlowFeatureExtractor
from src.pipeline.detection import DetectionPipeline
from src.prediction.mock_predictor import MockPredictor
from src.storage.database import Database


def _make_flow(
    flow_id: str,
    *,
    packet_count: int,
    byte_count: int,
    duration: float,
    src_ip: str = "10.0.0.1",
    dst_ip: str = "10.0.0.2",
    src_port: int = 5000,
    dst_port: int = 80,
    protocol: str = "tcp",
) -> FlowRecord:
    return FlowRecord(
        flow_id=flow_id,
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=src_port,
        dst_port=dst_port,
        protocol=protocol,
        start_time=100.0,
        end_time=100.0 + duration,
        duration=duration,
        packet_count=packet_count,
        byte_count=byte_count,
        summary_stats={
            "min_packet_size": 20,
            "max_packet_size": 100,
        },
    )


def _assert_dashboard_summary(database: Database, expected_flow_count: int, expected_packet_total: int, expected_byte_total: int, expected_normal: int, expected_attack: int, expected_alerts: int) -> None:
    dashboard = SecurityDashboard(database.db_path)
    summary = dashboard.get_summary()

    assert summary["flow_count"] == expected_flow_count
    assert summary["packet_count"] == expected_packet_total
    assert summary["byte_count"] == expected_byte_total
    assert summary["normal_count"] == expected_normal
    assert summary["attack_count"] == expected_attack
    assert len(summary["alerts"]) == expected_alerts


def test_end_to_end_normal_flow_pipeline_persists_and_dashboard_reads_it(tmp_path):
    database = Database(tmp_path / "normal_end_to_end.sqlite3")
    predictor = MockPredictor()
    pipeline = DetectionPipeline(predictor=predictor, database=database)

    flow = _make_flow("normal-flow", packet_count=0, byte_count=0, duration=0.0)

    feature_record = FlowFeatureExtractor().extract(flow)
    assert feature_record.metadata["flow_id"] == flow.flow_id

    prediction = predictor.predict(feature_record)
    assert prediction.prediction == "normal"

    result = pipeline.process_flow(flow)

    assert result["persisted"] is True
    assert result["alert"] is None

    stored_flow = database.get_flow(flow.flow_id)
    stored_prediction = database.get_prediction(flow.flow_id)

    assert stored_flow is not None
    assert stored_prediction is not None
    assert stored_flow.flow_id == flow.flow_id
    assert stored_prediction.flow_id == flow.flow_id
    assert stored_prediction.prediction == "normal"
    assert len(database.get_all_alerts()) == 0

    _assert_dashboard_summary(
        database,
        expected_flow_count=1,
        expected_packet_total=0,
        expected_byte_total=0,
        expected_normal=1,
        expected_attack=0,
        expected_alerts=0,
    )


def test_end_to_end_attack_flow_pipeline_persists_alert_and_dashboard_reads_it(tmp_path):
    database = Database(tmp_path / "attack_end_to_end.sqlite3")
    predictor = MockPredictor()
    pipeline = DetectionPipeline(predictor=predictor, database=database)

    flow = _make_flow("attack-flow", packet_count=200, byte_count=9000, duration=1.5)

    feature_record = FlowFeatureExtractor().extract(flow)
    assert feature_record.metadata["flow_id"] == flow.flow_id

    prediction = predictor.predict(feature_record)
    assert prediction.prediction == "attack"

    result = pipeline.process_flow(flow)

    assert result["persisted"] is True
    assert result["alert"] is not None
    assert result["alert"].flow_id == flow.flow_id
    assert result["alert"].prediction == "attack"

    stored_flow = database.get_flow(flow.flow_id)
    stored_prediction = database.get_prediction(flow.flow_id)
    alerts = database.get_all_alerts()

    assert stored_flow is not None
    assert stored_prediction is not None
    assert stored_prediction.flow_id == flow.flow_id
    assert stored_prediction.prediction == "attack"
    assert len(alerts) == 1
    assert alerts[0].flow_id == flow.flow_id

    _assert_dashboard_summary(
        database,
        expected_flow_count=1,
        expected_packet_total=200,
        expected_byte_total=9000,
        expected_normal=0,
        expected_attack=1,
        expected_alerts=1,
    )
