"""Safe demonstration of an attack prediction and alert persistence.

This script does NOT generate or send network traffic.

It takes one genuine N-BaIoT attack feature vector from the project's
test dataset, sends it through the real Random Forest predictor, creates
an AlertRecord when the prediction is Attack, and persists the result
to the project's real SQLite database.

This is intended for demonstration/testing only.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.core.contracts import AlertRecord, GenericFeatureRecord
from src.prediction.random_forest_predictor import RandomForestPredictor
from src.storage.database import Database


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATABASE_PATH = PROJECT_ROOT / "data" / "ids.sqlite3"
TEST_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"


def main() -> None:
    print("=" * 70)
    print("SAFE IDS ATTACK → ALERT DEMONSTRATION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # STEP 1 — Verify test dataset
    # ------------------------------------------------------------------

    print("\nSTEP 1 — LOADING N-BAIoT TEST DATASET")

    if not TEST_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_DATASET_PATH}"
        )

    df = pd.read_csv(TEST_DATASET_PATH)

    if "label" not in df.columns:
        raise ValueError("test.csv must contain a 'label' column")

    feature_columns = [
        column for column in df.columns
        if column != "label"
    ]

    print(f"Dataset: {TEST_DATASET_PATH}")
    print(f"Samples: {len(df)}")
    print(f"Features: {len(feature_columns)}")

    # ------------------------------------------------------------------
    # STEP 2 — Load Random Forest
    # ------------------------------------------------------------------

    print("\nSTEP 2 — LOADING RANDOM FOREST")

    predictor = RandomForestPredictor()
    model = predictor.load_model()

    print(f"Model: {type(model).__name__}")
    print(f"Model features: {len(model.feature_names_in_)}")

    if len(model.feature_names_in_) != 115:
        raise ValueError(
            "Random Forest does not expect 115 features"
        )

    print("✓ Random Forest loaded")

    # ------------------------------------------------------------------
    # STEP 3 — Find a genuine attack sample
    # ------------------------------------------------------------------

    print("\nSTEP 3 — FINDING GENUINE ATTACK SAMPLE")

    X = df[feature_columns]
    y = df["label"]

    predictions = model.predict(X)

    attack_indices = [
        index
        for index, prediction in zip(
            df.index,
            predictions,
        )
        if int(prediction) == 1
    ]

    if not attack_indices:
        raise RuntimeError(
            "No test sample is predicted as Attack"
        )

    attack_index = attack_indices[0]

    attack_features = X.loc[attack_index].copy()

    actual_label = int(y.loc[attack_index])
    model_prediction = int(predictions[attack_index])

    print(f"Selected sample index: {attack_index}")
    print(f"Actual label:          {actual_label}")
    print(f"Model prediction:      {model_prediction}")
    print(f"Feature count:         {len(attack_features)}")

    if actual_label != 1:
        raise AssertionError(
            "Selected sample is not an actual attack sample"
        )

    if model_prediction != 1:
        raise AssertionError(
            "Selected sample is not predicted as Attack"
        )

    print("✓ Genuine attack sample selected")

    # ------------------------------------------------------------------
    # STEP 4 — Create GenericFeatureRecord
    # ------------------------------------------------------------------

    print("\nSTEP 4 — CREATING N-BAIOT FEATURE RECORD")

    timestamp = time.time()
    packet_id = f"demo-attack-{attack_index}-{timestamp:.6f}"

    features = {
        column: float(attack_features[column])
        for column in feature_columns
    }

    feature_record = GenericFeatureRecord(
        feature_version="nbaiot-kitsune-v1",
        features=features,
        metadata={
            "flow_id": packet_id,
            "packet_id": packet_id,
            "timestamp": timestamp,
            "feature_source": "nbaiot_test_dataset",
            "demo": True,
            "dataset_sample_index": int(attack_index),
            "actual_label": actual_label,
        },
    )

    print(
        f"Feature version: {feature_record.feature_version}"
    )

    print(
        f"Feature count:   {len(feature_record.features)}"
    )

    print(
        f"Demo flow ID:    {packet_id}"
    )

    if len(feature_record.features) != 115:
        raise AssertionError(
            "Feature record does not contain exactly 115 features"
        )

    print("✓ GenericFeatureRecord created")

    # ------------------------------------------------------------------
    # STEP 5 — Run real Random Forest prediction
    # ------------------------------------------------------------------

    print("\nSTEP 5 — RUNNING REAL RANDOM FOREST PREDICTION")

    prediction = predictor.predict(feature_record)

    print(
        f"Prediction: {prediction.prediction}"
    )

    print(
        f"Probability: {prediction.probability}"
    )

    print(
        f"Flow ID: {prediction.flow_id}"
    )

    print(
        f"Model: {prediction.model_name}"
    )

    print(
        f"Source: {prediction.source}"
    )

    if prediction.prediction.lower() != "attack":
        raise AssertionError(
            "The genuine attack feature vector was not predicted as Attack"
        )

    print("✓ Random Forest produced ATTACK prediction")

    # ------------------------------------------------------------------
    # STEP 6 — Create AlertRecord
    # ------------------------------------------------------------------

    print("\nSTEP 6 — CREATING ALERT RECORD")

    alert = AlertRecord(
        alert_id=f"alert-{prediction.flow_id}-{int(timestamp)}",
        flow_id=prediction.flow_id,
        timestamp=prediction.timestamp,
        prediction=prediction.prediction,
        probability=prediction.probability,
        status="new",
    )

    print(f"Alert ID:    {alert.alert_id}")
    print(f"Flow ID:     {alert.flow_id}")
    print(f"Prediction:  {alert.prediction}")
    print(f"Probability: {alert.probability}")
    print(f"Status:      {alert.status}")

    print("✓ AlertRecord created")

    # ------------------------------------------------------------------
    # STEP 7 — Persist to actual project database
    # ------------------------------------------------------------------

    print("\nSTEP 7 — PERSISTING TO PROJECT SQLITE DATABASE")

    database = Database(DATABASE_PATH)
    database.initialize()

    database.insert_feature_record(feature_record)
    database.insert_prediction(prediction)
    database.insert_alert(alert)

    print(f"Database: {DATABASE_PATH}")
    print("✓ Feature record persisted")
    print("✓ Prediction persisted")
    print("✓ Alert persisted")

    # ------------------------------------------------------------------
    # STEP 8 — Verify alert from SQLite
    # ------------------------------------------------------------------

    print("\nSTEP 8 — VERIFYING ALERT FROM SQLITE")

    stored_alert = database.get_alert(alert.alert_id)

    if stored_alert is None:
        raise AssertionError(
            "AlertRecord was not found in SQLite"
        )

    if stored_alert.prediction.lower() != "attack":
        raise AssertionError(
            "Stored prediction is not Attack"
        )

    if stored_alert.flow_id != prediction.flow_id:
        raise AssertionError(
            "Stored alert flow_id does not match prediction flow_id"
        )

    print("✓ AlertRecord found in SQLite")
    print(
        f"Stored prediction: {stored_alert.prediction}"
    )
    print(
        f"Stored probability: {stored_alert.probability}"
    )
    print(
        f"Stored status:      {stored_alert.status}"
    )

    # ------------------------------------------------------------------
    # FINAL VALIDATION
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)

    print("✓ Genuine N-BaIoT attack sample selected")
    print("✓ 115-feature schema validated")
    print("✓ Random Forest predicted Attack")
    print("✓ AlertRecord generated")
    print("✓ Prediction persisted to SQLite")
    print("✓ Alert persisted to SQLite")
    print("✓ Stored alert verified")

    print("\n" + "=" * 70)
    print("SUCCESS!")
    print("Safe Attack → Alert → SQLite demonstration passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()