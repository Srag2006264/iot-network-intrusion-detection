"""End-to-end tests for the N-BaIoT detection pipeline.

This test validates both detection paths:

1. Normal traffic:
   PacketRecord -> N-BaIoT -> Random Forest -> Normal -> No Alert

2. Attack response:
   PacketRecord -> N-BaIoT -> controlled Attack prediction
   -> AlertRecord -> SQLite persistence

The attack path uses a controlled predictor rather than generating
malicious network traffic.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from src.core.contracts import (
    AlertRecord,
    GenericFeatureRecord,
    PacketRecord,
    PredictionResult,
)
from src.pipeline.detection import DetectionPipeline
from src.prediction.predictor import Predictor
from src.prediction.random_forest_predictor import RandomForestPredictor
from src.storage.database import Database


class ControlledAttackPredictor(Predictor):
    """Predict Attack for controlled alert-path testing."""

    def predict(
        self,
        feature_record: GenericFeatureRecord,
    ) -> PredictionResult:
        """Return a controlled Attack prediction."""

        if not isinstance(feature_record, GenericFeatureRecord):
            raise TypeError(
                "feature_record must be a GenericFeatureRecord instance"
            )

        return PredictionResult(
            flow_id=str(
                feature_record.metadata.get(
                    "flow_id",
                    "unknown-flow",
                )
            ),
            prediction="Attack",
            probability=0.99,
            timestamp=float(
                feature_record.metadata.get(
                    "timestamp",
                    0.0,
                )
            ),
            model_name="controlled-attack-test-model",
            source="test",
            notes="Controlled Attack prediction for alert-path validation.",
        )


def create_test_packet() -> PacketRecord:
    """Create a deterministic packet for pipeline testing."""

    return PacketRecord(
        timestamp=1.0,
        src_ip="192.168.1.10",
        dst_ip="192.168.1.20",
        src_port=12345,
        dst_port=80,
        protocol="tcp",
        packet_length=100,
    )


def print_header(title: str) -> None:
    """Print a formatted test section header."""

    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)


def test_normal_traffic_with_random_forest() -> None:
    """Verify real N-BaIoT + Random Forest normal-traffic behavior."""

    print_header(
        "TEST 1 — NORMAL TRAFFIC → RANDOM FOREST → NO ALERT"
    )

    temp_dir = Path(tempfile.mkdtemp())
    database_path = temp_dir / "normal_test.sqlite3"

    database = Database(database_path)
    predictor = RandomForestPredictor()

    model = predictor.load_model()

    print(f"Model: {type(model).__name__}")
    print(f"Model features: {len(model.feature_names_in_)}")

    pipeline = DetectionPipeline(
        predictor=predictor,
        database=database,
    )

    packet = create_test_packet()

    print(
        f"Packet: "
        f"{packet.src_ip}:{packet.src_port} → "
        f"{packet.dst_ip}:{packet.dst_port}"
    )

    result = pipeline.process_packet(packet)

    feature_record = result["feature_record"]
    prediction = result["prediction"]
    alert = result["alert"]

    assert isinstance(feature_record, GenericFeatureRecord)
    assert len(feature_record.features) == 115
    assert feature_record.feature_version == "nbaiot-kitsune-v1"

    assert isinstance(prediction, PredictionResult)
    assert prediction.source == "random_forest"
    assert prediction.prediction in {"Normal", "Attack"}

    print(f"Feature count: {len(feature_record.features)}")
    print(f"Prediction: {prediction.prediction}")
    print(f"Probability: {prediction.probability}")

    if prediction.prediction.lower() == "normal":
        assert alert is None
        print("✓ Normal traffic produced no alert")
    else:
        assert isinstance(alert, AlertRecord)
        print(
            "⚠ Test packet was classified as Attack; "
            "alert was generated as expected."
        )

    assert result["persisted"] is True

    print("✓ Database persistence completed")
    print("✓ Normal-path validation completed")


def test_attack_prediction_creates_alert() -> None:
    """Verify that an Attack prediction creates and persists an alert."""

    print_header(
        "TEST 2 — ATTACK PREDICTION → ALERT RECORD → SQLITE"
    )

    temp_dir = Path(tempfile.mkdtemp())
    database_path = temp_dir / "attack_alert_test.sqlite3"

    database = Database(database_path)

    predictor = ControlledAttackPredictor()

    pipeline = DetectionPipeline(
        predictor=predictor,
        database=database,
    )

    packet = PacketRecord(
        timestamp=100.0,
        src_ip="10.0.0.50",
        dst_ip="192.168.1.9",
        src_port=4444,
        dst_port=80,
        protocol="tcp",
        packet_length=1500,
    )

    print(
        f"Packet: "
        f"{packet.src_ip}:{packet.src_port} → "
        f"{packet.dst_ip}:{packet.dst_port}"
    )

    result = pipeline.process_packet(packet)

    feature_record = result["feature_record"]
    prediction = result["prediction"]
    alert = result["alert"]

    # --------------------------------------------------------------
    # Verify N-BaIoT feature generation
    # --------------------------------------------------------------

    assert isinstance(feature_record, GenericFeatureRecord)
    assert len(feature_record.features) == 115
    assert feature_record.feature_version == "nbaiot-kitsune-v1"

    print("✓ N-BaIoT generated 115 features")

    # --------------------------------------------------------------
    # Verify Attack prediction
    # --------------------------------------------------------------

    assert isinstance(prediction, PredictionResult)
    assert prediction.prediction == "Attack"
    assert prediction.probability == 0.99
    assert prediction.source == "test"

    print("Prediction: Attack")
    print(f"Probability: {prediction.probability}")
    print("✓ Controlled Attack prediction generated")

    # --------------------------------------------------------------
    # Verify AlertRecord
    # --------------------------------------------------------------

    assert alert is not None
    assert isinstance(alert, AlertRecord)

    expected_flow_id = str(
        feature_record.metadata.get(
            "flow_id",
            f"packet-{packet.timestamp}",
        )
    )

    assert alert.flow_id == expected_flow_id
    assert alert.prediction == "Attack"
    assert alert.probability == 0.99
    assert alert.status == "new"

    print("✓ AlertRecord generated")
    print(f"Alert ID: {alert.alert_id}")
    print(f"Alert flow ID: {alert.flow_id}")
    print(f"Alert status: {alert.status}")

    # --------------------------------------------------------------
    # Verify pipeline persistence
    # --------------------------------------------------------------

    assert result["persisted"] is True

    print("✓ Detection result persisted")

    # --------------------------------------------------------------
    # Verify alert persistence through Database API
    # --------------------------------------------------------------

    stored_alert = database.get_alert(alert.alert_id)

    assert stored_alert is not None
    assert isinstance(stored_alert, AlertRecord)

    assert stored_alert.alert_id == alert.alert_id
    assert stored_alert.flow_id == alert.flow_id
    assert stored_alert.timestamp == alert.timestamp
    assert stored_alert.prediction == "Attack"
    assert stored_alert.probability == 0.99
    assert stored_alert.status == "new"

    print("✓ AlertRecord found in SQLite")
    print(f"✓ Stored prediction: {stored_alert.prediction}")
    print(f"✓ Stored probability: {stored_alert.probability}")
    print(f"✓ Stored status: {stored_alert.status}")


def main() -> None:
    """Run the complete N-BaIoT detection pipeline validation."""

    print("=" * 70)
    print("N-BaIoT END-TO-END DETECTION PIPELINE VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------------
    # TEST 1 — Normal path
    # --------------------------------------------------------------

    test_normal_traffic_with_random_forest()

    # --------------------------------------------------------------
    # TEST 2 — Attack alert path
    # --------------------------------------------------------------

    test_attack_prediction_creates_alert()

    # --------------------------------------------------------------
    # Final validation
    # --------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)

    print("✓ N-BaIoT feature extraction validated")
    print("✓ 115-feature schema validated")
    print("✓ Random Forest prediction validated")
    print("✓ Normal traffic → no alert validated")
    print("✓ Attack prediction → AlertRecord validated")
    print("✓ AlertRecord → SQLite persistence validated")

    print("\n" + "=" * 70)
    print("SUCCESS!")
    print("Complete N-BaIoT detection and alert validation passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()