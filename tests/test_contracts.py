"""Tests for shared project contracts."""

from src.core.contracts import (
    AlertRecord,
    FlowRecord,
    GenericFeatureRecord,
    PacketRecord,
    PredictionResult,
)


def test_packet_record_can_be_constructed():
    record = PacketRecord(
        timestamp=1.5,
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        src_port=1234,
        dst_port=80,
        protocol="tcp",
        packet_length=150,
        flags="S",
    )

    assert record.timestamp == 1.5
    assert record.src_ip == "10.0.0.1"
    assert record.dst_ip == "10.0.0.2"
    assert record.src_port == 1234
    assert record.dst_port == 80
    assert record.protocol == "tcp"
    assert record.packet_length == 150
    assert record.flags == "S"


def test_flow_record_can_be_constructed():
    record = FlowRecord(
        flow_id="flow-001",
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        src_port=1234,
        dst_port=80,
        protocol="tcp",
        start_time=1.0,
        end_time=2.0,
        duration=1.0,
        packet_count=10,
        byte_count=500,
        summary_stats={"mean_len": 50.0},
    )

    assert record.flow_id == "flow-001"
    assert record.src_ip == "10.0.0.1"
    assert record.dst_ip == "10.0.0.2"
    assert record.packet_count == 10
    assert record.summary_stats["mean_len"] == 50.0


def test_generic_feature_record_can_be_constructed():
    record = GenericFeatureRecord(
        feature_version="1.0",
        features={"packet_count": 10, "byte_count": 500, "is_active": True},
        metadata={"flow_id": "flow-001", "timestamp": 1.0},
    )

    assert record.feature_version == "1.0"
    assert record.features["packet_count"] == 10
    assert record.features["is_active"] is True
    assert record.metadata["flow_id"] == "flow-001"


def test_prediction_result_can_be_constructed():
    result = PredictionResult(
        flow_id="flow-001",
        prediction="normal",
        probability=0.9,
        timestamp=1.0,
        model_name="mock-model",
        source="mock",
        notes="development-only",
    )

    assert result.flow_id == "flow-001"
    assert result.prediction == "normal"
    assert result.probability == 0.9
    assert result.model_name == "mock-model"
    assert result.source == "mock"
    assert result.notes == "development-only"


def test_alert_record_can_be_constructed():
    alert = AlertRecord(
        alert_id="alert-001",
        flow_id="flow-001",
        timestamp=2.0,
        prediction="attack",
        probability=0.85,
        status="new",
    )

    assert alert.alert_id == "alert-001"
    assert alert.flow_id == "flow-001"
    assert alert.prediction == "attack"
    assert alert.status == "new"
