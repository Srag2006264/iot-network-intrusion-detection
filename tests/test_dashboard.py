"""Tests for the Streamlit dashboard data access layer."""

from __future__ import annotations

from pathlib import Path

from src.core.contracts import AlertRecord, FlowRecord, PredictionResult
from src.dashboard.app import SecurityDashboard
from src.storage.database import Database


def _seed_database(database: Database) -> None:
    database.initialize()

    flows = [
        FlowRecord(
            flow_id="flow-1",
            src_ip="10.0.0.1",
            dst_ip="10.0.0.2",
            src_port=5000,
            dst_port=80,
            protocol="tcp",
            start_time=1.0,
            end_time=2.0,
            duration=1.0,
            packet_count=10,
            byte_count=150,
            summary_stats={"min_packet_size": 10, "max_packet_size": 40},
        ),
        FlowRecord(
            flow_id="flow-2",
            src_ip="10.0.0.3",
            dst_ip="10.0.0.4",
            src_port=5001,
            dst_port=80,
            protocol="udp",
            start_time=3.0,
            end_time=5.0,
            duration=2.0,
            packet_count=8,
            byte_count=220,
            summary_stats={"min_packet_size": 20, "max_packet_size": 50},
        ),
    ]

    for flow in flows:
        database.insert_flow(flow)

    predictions = [
        PredictionResult(
            flow_id="flow-1",
            prediction="normal",
            probability=0.88,
            timestamp=10.0,
            model_name="mock-model",
            source="mock",
            notes="normal",
        ),
        PredictionResult(
            flow_id="flow-2",
            prediction="attack",
            probability=0.96,
            timestamp=11.0,
            model_name="mock-model",
            source="mock",
            notes="attack",
        ),
    ]

    for prediction in predictions:
        database.insert_prediction(prediction)

    alerts = [
        AlertRecord(
            alert_id="alert-1",
            flow_id="flow-2",
            timestamp=12.0,
            prediction="attack",
            probability=0.96,
            status="new",
        )
    ]

    for alert in alerts:
        database.insert_alert(alert)


def test_dashboard_can_read_flow_records(tmp_path):
    database = Database(tmp_path / "dashboard.sqlite3")
    _seed_database(database)

    dashboard = SecurityDashboard(database.db_path)
    flows = dashboard.get_flow_records()

    assert len(flows) == 2
    assert flows[0].flow_id == "flow-1"


def test_dashboard_can_read_predictions(tmp_path):
    database = Database(tmp_path / "dashboard.sqlite3")
    _seed_database(database)

    dashboard = SecurityDashboard(database.db_path)
    predictions = dashboard.get_predictions()

    assert len(predictions) == 2
    assert predictions[1].prediction == "attack"


def test_dashboard_can_read_alerts(tmp_path):
    database = Database(tmp_path / "dashboard.sqlite3")
    _seed_database(database)

    dashboard = SecurityDashboard(database.db_path)
    alerts = dashboard.get_alerts()

    assert len(alerts) == 1
    assert alerts[0].flow_id == "flow-2"


def test_dashboard_handles_empty_database(tmp_path):
    dashboard = SecurityDashboard(tmp_path / "empty.sqlite3")
    summary = dashboard.get_summary()

    assert summary["flow_count"] == 0
    assert summary["packet_count"] == 0
    assert summary["byte_count"] == 0
    assert summary["normal_count"] == 0
    assert summary["attack_count"] == 0


def test_dashboard_correctly_counts_normal_predictions(tmp_path):
    database = Database(tmp_path / "dashboard.sqlite3")
    _seed_database(database)

    dashboard = SecurityDashboard(database.db_path)
    summary = dashboard.get_summary()

    assert summary["normal_count"] == 1


def test_dashboard_correctly_counts_attack_predictions(tmp_path):
    database = Database(tmp_path / "dashboard.sqlite3")
    _seed_database(database)

    dashboard = SecurityDashboard(database.db_path)
    summary = dashboard.get_summary()

    assert summary["attack_count"] == 1


def test_dashboard_calculates_total_traffic_totals(tmp_path):
    database = Database(tmp_path / "dashboard.sqlite3")
    _seed_database(database)

    dashboard = SecurityDashboard(database.db_path)
    summary = dashboard.get_summary()

    assert summary["flow_count"] == 2
    assert summary["packet_count"] == 18
    assert summary["byte_count"] == 370


def test_dashboard_handles_missing_optional_prediction_confidence(tmp_path):
    database = Database(tmp_path / "dashboard.sqlite3")
    database.initialize()

    database.insert_prediction(
        PredictionResult(
            flow_id="flow-missing-confidence",
            prediction="normal",
            probability=None,
            timestamp=50.0,
            model_name="mock-model",
            source="mock",
            notes="no-confidence",
        )
    )

    dashboard = SecurityDashboard(database.db_path)
    predictions = dashboard.get_predictions()

    assert len(predictions) == 1
    assert predictions[0].probability is None


def test_dashboard_uses_existing_database_layer_not_duplicate_schema_logic(tmp_path):
    database = Database(tmp_path / "dashboard.sqlite3")
    _seed_database(database)

    dashboard = SecurityDashboard(database.db_path)
    flows = dashboard.get_flow_records()
    predictions = dashboard.get_predictions()
    alerts = dashboard.get_alerts()

    assert len(flows) == 2
    assert len(predictions) == 2
    assert len(alerts) == 1


def test_dashboard_does_not_require_scapy_or_random_forest(tmp_path):
    database = Database(tmp_path / "dashboard.sqlite3")
    _seed_database(database)

    dashboard = SecurityDashboard(database.db_path)
    summary = dashboard.get_summary()

    assert summary["flow_count"] == 2
    assert summary["attack_count"] == 1


def test_dashboard_data_access_is_deterministic_for_fixed_database_state(tmp_path):
    database = Database(tmp_path / "dashboard.sqlite3")
    _seed_database(database)

    first = SecurityDashboard(database.db_path).get_summary()
    second = SecurityDashboard(database.db_path).get_summary()

    assert first == second


def test_dashboard_uses_database_path_configuration(tmp_path):
    db_path = tmp_path / "configured.sqlite3"
    database = Database(db_path)
    _seed_database(database)

    dashboard = SecurityDashboard(db_path)
    assert len(dashboard.get_flow_records()) == 2
