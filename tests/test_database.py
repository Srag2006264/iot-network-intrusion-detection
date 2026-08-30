"""Tests for the SQLite persistence layer."""

import sqlite3

import pytest

from src.core.contracts import AlertRecord, FlowRecord, GenericFeatureRecord, PredictionResult
from src.storage.database import Database


@pytest.fixture
def temp_database(tmp_path):
    db_path = tmp_path / "test_database.sqlite3"
    database = Database(db_path)
    database.initialize()
    yield database
    database.close()


def test_database_can_be_initialized(temp_database):
    assert temp_database is not None


def test_required_tables_are_created(temp_database):
    with sqlite3.connect(temp_database.db_path) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()

    table_names = {row[0] for row in tables}
    assert {"flows", "predictions", "alerts", "feature_records"}.issubset(table_names)


def test_flow_record_can_be_inserted_and_retrieved(temp_database):
    flow = FlowRecord(
        flow_id="flow-001",
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        src_port=5000,
        dst_port=80,
        protocol="tcp",
        start_time=1.0,
        end_time=2.0,
        duration=1.0,
        packet_count=10,
        byte_count=500,
        summary_stats={"min_packet_size": 10, "max_packet_size": 90},
    )

    temp_database.insert_flow(flow)
    stored = temp_database.get_flow("flow-001")

    assert stored is not None
    assert stored.flow_id == "flow-001"
    assert stored.src_ip == "10.0.0.1"
    assert stored.packet_count == 10


def test_generic_feature_record_can_be_stored_and_retrieved(temp_database):
    features = GenericFeatureRecord(
        feature_version="generic-flow-v1",
        features={"packet_count": 10, "byte_count": 500, "duration": 2.0},
        metadata={"flow_id": "flow-002", "protocol": "tcp"},
    )

    temp_database.insert_feature_record(features)
    stored = temp_database.get_feature_record("flow-002")

    assert stored is not None
    assert stored.feature_version == "generic-flow-v1"
    assert stored.features["packet_count"] == 10
    assert stored.metadata["flow_id"] == "flow-002"


def test_prediction_result_can_be_stored_and_retrieved(temp_database):
    prediction = PredictionResult(
        flow_id="flow-003",
        prediction="attack",
        probability=0.85,
        timestamp=100.0,
        model_name="mock-model",
        source="mock",
        notes="development-only",
    )

    temp_database.insert_prediction(prediction)
    stored = temp_database.get_prediction("flow-003")

    assert stored is not None
    assert stored.prediction == "attack"
    assert stored.model_name == "mock-model"


def test_alert_record_can_be_stored_and_retrieved(temp_database):
    alert = AlertRecord(
        alert_id="alert-001",
        flow_id="flow-003",
        timestamp=101.0,
        prediction="attack",
        probability=0.85,
        status="new",
    )

    temp_database.insert_alert(alert)
    stored = temp_database.get_alert("alert-001")

    assert stored is not None
    assert stored.flow_id == "flow-003"
    assert stored.status == "new"


def test_multiple_records_can_be_stored(temp_database):
    first_flow = FlowRecord(
        flow_id="flow-a",
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        packet_count=3,
        byte_count=100,
        duration=1.0,
    )
    second_flow = FlowRecord(
        flow_id="flow-b",
        src_ip="10.0.0.3",
        dst_ip="10.0.0.4",
        packet_count=4,
        byte_count=200,
        duration=2.0,
    )

    temp_database.insert_flow(first_flow)
    temp_database.insert_flow(second_flow)

    flows = temp_database.get_all_flows()
    assert len(flows) == 2


def test_database_initialization_is_safe_to_call_more_than_once(temp_database):
    temp_database.initialize()
    temp_database.initialize()

    with sqlite3.connect(temp_database.db_path) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()

    assert {"flows", "predictions", "alerts", "feature_records"}.issubset({row[0] for row in tables})


def test_database_uses_parameterized_sql_not_string_interpolation(temp_database):
    flow = FlowRecord(
        flow_id="flow-param-test",
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        packet_count=1,
        byte_count=50,
        duration=1.0,
    )

    temp_database.insert_flow(flow)
    stored = temp_database.get_flow("flow-param-test")
    assert stored is not None
    assert stored.flow_id == "flow-param-test"


def test_data_persists_after_closing_and_reopening_database_connection(tmp_path):
    db_path = tmp_path / "persist.sqlite3"
    database = Database(db_path)
    database.initialize()

    flow = FlowRecord(
        flow_id="flow-persist",
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        packet_count=2,
        byte_count=80,
        duration=1.5,
    )
    database.insert_flow(flow)
    database.close()

    reopened = Database(db_path)
    stored = reopened.get_flow("flow-persist")

    assert stored is not None
    assert stored.flow_id == "flow-persist"
    assert stored.packet_count == 2


def test_missing_optional_values_are_handled_correctly(temp_database):
    prediction = PredictionResult(
        flow_id="flow-optional",
        prediction="normal",
        probability=None,
        timestamp=200.0,
        model_name="mock-model",
        source="mock",
        notes=None,
    )

    temp_database.insert_prediction(prediction)
    stored = temp_database.get_prediction("flow-optional")

    assert stored is not None
    assert stored.probability is None
    assert stored.notes is None


def test_flow_and_prediction_and_alert_records_can_be_retrieved_as_lists(temp_database):
    flow = FlowRecord(
        flow_id="flow-list",
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        packet_count=5,
        byte_count=250,
        duration=2.0,
    )
    prediction = PredictionResult(
        flow_id="flow-list",
        prediction="normal",
        probability=0.2,
        timestamp=300.0,
        model_name="mock-model",
        source="mock",
        notes="ok",
    )
    alert = AlertRecord(
        alert_id="alert-list",
        flow_id="flow-list",
        timestamp=301.0,
        prediction="normal",
        probability=0.2,
        status="new",
    )

    temp_database.insert_flow(flow)
    temp_database.insert_prediction(prediction)
    temp_database.insert_alert(alert)

    assert len(temp_database.get_all_flows()) == 1
    assert len(temp_database.get_all_predictions()) == 1
    assert len(temp_database.get_all_alerts()) == 1
