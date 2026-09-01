"""End-to-end test for the N-BaIoT detection pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

from src.core.contracts import PacketRecord, PredictionResult
from src.pipeline.detection import DetectionPipeline
from src.prediction.random_forest_predictor import RandomForestPredictor
from src.storage.database import Database


def main() -> None:
    print("=" * 70)
    print("N-BaIoT END-TO-END DETECTION PIPELINE TEST")
    print("=" * 70)

    # --------------------------------------------------------------
    # STEP 1 — Create temporary database
    # --------------------------------------------------------------

    print("\nSTEP 1 — CREATING TEST DATABASE")

    temp_dir = Path(tempfile.mkdtemp())
    database_path = temp_dir / "nbaiot_pipeline_test.sqlite3"

    database = Database(database_path)

    print(f"Database: {database_path}")
    print("✓ Test database created")

    # --------------------------------------------------------------
    # STEP 2 — Load Random Forest
    # --------------------------------------------------------------

    print("\nSTEP 2 — LOADING RANDOM FOREST")

    predictor = RandomForestPredictor()

    model = predictor.load_model()

    print(f"Model: {type(model).__name__}")
    print(f"Model features: {len(model.feature_names_in_)}")
    print("✓ Random Forest loaded")

    # --------------------------------------------------------------
    # STEP 3 — Create detection pipeline
    # --------------------------------------------------------------

    print("\nSTEP 3 — CREATING DETECTION PIPELINE")

    pipeline = DetectionPipeline(
        predictor=predictor,
        database=database,
    )

    print("✓ DetectionPipeline created")
    print(
        f"N-BaIoT features: "
        f"{pipeline.nbaiot_feature_extractor.get_num_features()}"
    )

    # --------------------------------------------------------------
    # STEP 4 — Create test packet
    # --------------------------------------------------------------

    print("\nSTEP 4 — CREATING TEST PACKET")

    packet = PacketRecord(
        timestamp=1.0,
        src_ip="192.168.1.10",
        dst_ip="192.168.1.20",
        src_port=12345,
        dst_port=80,
        protocol="tcp",
        packet_length=100,
    )

    print(f"Source: {packet.src_ip}:{packet.src_port}")
    print(f"Destination: {packet.dst_ip}:{packet.dst_port}")
    print(f"Protocol: {packet.protocol}")
    print(f"Packet length: {packet.packet_length}")
    print("✓ PacketRecord created")

    # --------------------------------------------------------------
    # STEP 5 — Process packet
    # --------------------------------------------------------------

    print("\nSTEP 5 — PROCESSING PACKET")

    result = pipeline.process_packet(packet)

    print("✓ Packet processed")

    # --------------------------------------------------------------
    # STEP 6 — Verify feature record
    # --------------------------------------------------------------

    print("\nSTEP 6 — VERIFYING N-BaIoT FEATURES")

    feature_record = result["feature_record"]

    print(f"Feature version: {feature_record.feature_version}")
    print(f"Feature count: {len(feature_record.features)}")

    assert len(feature_record.features) == 115
    assert feature_record.feature_version == "nbaiot-kitsune-v1"

    print("✓ 115 N-BaIoT features generated")

    # --------------------------------------------------------------
    # STEP 7 — Verify prediction
    # --------------------------------------------------------------

    print("\nSTEP 7 — VERIFYING RANDOM FOREST PREDICTION")

    prediction = result["prediction"]

    assert isinstance(prediction, PredictionResult)

    print(f"Prediction: {prediction.prediction}")
    print(f"Probability: {prediction.probability}")
    print(f"Model: {prediction.model_name}")
    print(f"Source: {prediction.source}")

    assert prediction.source == "random_forest"
    assert prediction.prediction in {"Normal", "Attack"}

    print("✓ Valid PredictionResult generated")

    # --------------------------------------------------------------
    # STEP 8 — Verify alert behavior
    # --------------------------------------------------------------

    print("\nSTEP 8 — VERIFYING ALERT BEHAVIOR")

    alert = result["alert"]

    if prediction.prediction.lower() == "attack":
        assert alert is not None
        print("Prediction is ATTACK")
        print("✓ AlertRecord generated")
    else:
        assert alert is None
        print("Prediction is NORMAL")
        print("✓ No alert generated for normal traffic")

    # --------------------------------------------------------------
    # STEP 9 — Verify persistence
    # --------------------------------------------------------------

    print("\nSTEP 9 — VERIFYING DATABASE PERSISTENCE")

    assert result["persisted"] is True

    print("✓ Detection result persisted")

    # --------------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)

    print("✓ PacketRecord accepted")
    print("✓ N-BaIoT extractor generated 115 features")
    print("✓ GenericFeatureRecord created")
    print("✓ Random Forest prediction generated")
    print("✓ PredictionResult created")
    print("✓ Database persistence completed")

    if prediction.prediction.lower() == "attack":
        print("✓ Attack alert generated")
    else:
        print("✓ Normal traffic produced no alert")

    print("\n" + "=" * 70)
    print("SUCCESS!")
    print("Complete N-BaIoT detection pipeline test passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()