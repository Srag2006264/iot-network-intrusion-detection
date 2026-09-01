"""Integration test for N-BaIoT feature extraction and Random Forest."""

from __future__ import annotations

from pathlib import Path

import joblib

from src.core.contracts import PacketRecord
from src.network.nbaiot_features import NBAIoTFeatureExtractor


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "ml" / "model" / "random_forest.joblib"


def main() -> None:
    print("=" * 70)
    print("N-BaIoT → RANDOM FOREST INTEGRATION TEST")
    print("=" * 70)

    # ---------------------------------------------------------------
    # STEP 1 — Load model
    # ---------------------------------------------------------------

    print("\nSTEP 1 — LOADING RANDOM FOREST")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Random Forest model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    print("✓ Random Forest loaded.")
    print("Model:", MODEL_PATH)

    # ---------------------------------------------------------------
    # STEP 2 — Verify model schema
    # ---------------------------------------------------------------

    print("\nSTEP 2 — VERIFYING MODEL")

    expected_features = getattr(model, "n_features_in_", None)

    if expected_features != 115:
        raise ValueError(
            f"Expected model to require 115 features, "
            f"but model requires {expected_features}"
        )

    print("✓ Model expects exactly 115 features.")

    # ---------------------------------------------------------------
    # STEP 3 — Create feature extractor
    # ---------------------------------------------------------------

    print("\nSTEP 3 — CREATING N-BaIoT FEATURE EXTRACTOR")

    extractor = NBAIoTFeatureExtractor()

    feature_count = extractor.get_num_features()

    print("Extractor feature count:", feature_count)

    if feature_count != 115:
        raise ValueError(
            f"Expected 115 features, got {feature_count}"
        )

    print("✓ Extractor produces exactly 115 features.")

    # ---------------------------------------------------------------
    # STEP 4 — Create test packet
    # ---------------------------------------------------------------

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

    print("✓ PacketRecord created.")

    # ---------------------------------------------------------------
    # STEP 5 — Extract N-BaIoT features
    # ---------------------------------------------------------------

    print("\nSTEP 5 — EXTRACTING N-BaIoT FEATURES")

    features = extractor.extract_vector(packet)

    print("Feature vector length:", len(features))

    if len(features) != 115:
        raise ValueError(
            f"Expected 115 extracted features, got {len(features)}"
        )

    print("✓ 115 features extracted.")

    # ---------------------------------------------------------------
    # STEP 6 — Verify numeric values
    # ---------------------------------------------------------------

    print("\nSTEP 6 — VERIFYING FEATURE VALUES")

    non_numeric = [
        value
        for value in features
        if not isinstance(value, (int, float))
    ]

    if non_numeric:
        raise ValueError(
            f"Found non-numeric feature values: {non_numeric[:5]}"
        )

    print("✓ All features are numeric.")

    # ---------------------------------------------------------------
    # STEP 7 — Run Random Forest prediction
    # ---------------------------------------------------------------

    print("\nSTEP 7 — RUNNING RANDOM FOREST PREDICTION")

    prediction = model.predict([features])[0]

    print("Raw prediction:", prediction)

    if prediction not in [0, 1]:
        raise ValueError(
            f"Unexpected prediction: {prediction}"
        )

    label = "Normal" if prediction == 0 else "Attack"

    print("Predicted label:", label)

    # ---------------------------------------------------------------
    # STEP 8 — Probability
    # ---------------------------------------------------------------

    print("\nSTEP 8 — CALCULATING CONFIDENCE")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([features])[0]

        normal_probability = float(probabilities[0])
        attack_probability = float(probabilities[1])

        print(
            f"Normal probability: {normal_probability:.6f}"
        )

        print(
            f"Attack probability: {attack_probability:.6f}"
        )

        confidence = float(max(probabilities))

        print(
            f"Prediction confidence: {confidence * 100:.2f}%"
        )

    # ---------------------------------------------------------------
    # FINAL VALIDATION
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)

    print("✓ Random Forest loaded")
    print("✓ Model expects 115 features")
    print("✓ N-BaIoT extractor produces 115 features")
    print("✓ Feature vector is numeric")
    print("✓ Feature vector accepted by Random Forest")
    print("✓ Prediction generated")
    print("✓ Prediction is valid")

    print("\n" + "=" * 70)
    print("SUCCESS!")
    print("N-BaIoT → Random Forest integration test passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()