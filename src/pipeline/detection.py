"""Detection pipeline for network flow and packet-level ML detection."""

from __future__ import annotations

from typing import Any

from src.core.contracts import (
    AlertRecord,
    FlowRecord,
    GenericFeatureRecord,
    PacketRecord,
    PredictionResult,
)
from src.network.features import FlowFeatureExtractor
from src.network.nbaiot_features import NBAIoTFeatureExtractor
from src.prediction.predictor import Predictor
from src.storage.database import Database


class DetectionPipeline:
    """Connect feature extraction, prediction, and persistence.

    The pipeline supports two paths:

    1. Existing flow-level path:
        FlowRecord
            -> FlowFeatureExtractor
            -> Predictor
            -> Database

    2. N-BaIoT packet-level ML path:
        PacketRecord
            -> NBAIoTFeatureExtractor
            -> GenericFeatureRecord
            -> Predictor
            -> Database
            -> AlertRecord when prediction is Attack

    The N-BaIoT path is model-agnostic at the pipeline level.
    A RandomForestPredictor can be supplied through the Predictor interface.
    """

    def __init__(
        self,
        feature_extractor: FlowFeatureExtractor | None = None,
        predictor: Predictor | None = None,
        database: Database | None = None,
        nbaiot_feature_extractor: NBAIoTFeatureExtractor | None = None,
    ) -> None:
        # Existing flow-level feature extractor.
        self.feature_extractor = feature_extractor or FlowFeatureExtractor()

        # N-BaIoT/Kitsune-compatible 115-feature extractor.
        self.nbaiot_feature_extractor = (
            nbaiot_feature_extractor
            or NBAIoTFeatureExtractor()
        )

        self.predictor = predictor
        self.database = database

    # ------------------------------------------------------------------
    # Existing flow-level detection path
    # ------------------------------------------------------------------

    def process_flow(self, flow: FlowRecord) -> dict[str, Any]:
        """Process one FlowRecord through the existing pipeline."""

        if not isinstance(flow, FlowRecord):
            raise TypeError("flow must be a FlowRecord instance")

        if self.predictor is None:
            raise ValueError(
                "predictor is required for detection processing"
            )

        if self.database is None:
            raise ValueError(
                "database is required for detection processing"
            )

        feature_record = self.feature_extractor.extract(flow)

        prediction = self.predictor.predict(feature_record)

        if not isinstance(prediction, PredictionResult):
            raise TypeError(
                "predictor must return a PredictionResult"
            )

        self.database.initialize()

        self.database.insert_flow(flow)
        self.database.insert_feature_record(feature_record)
        self.database.insert_prediction(prediction)

        alert: AlertRecord | None = None

        if prediction.prediction.lower() == "attack":
            alert = AlertRecord(
                alert_id=(
                    f"alert-{flow.flow_id}-"
                    f"{int(prediction.timestamp)}"
                ),
                flow_id=flow.flow_id,
                timestamp=float(prediction.timestamp),
                prediction=prediction.prediction,
                probability=prediction.probability,
                status="new",
            )

            self.database.insert_alert(alert)

        return {
            "flow": flow,
            "feature_record": feature_record,
            "prediction": prediction,
            "alert": alert,
            "persisted": True,
        }

    # ------------------------------------------------------------------
    # N-BaIoT packet-level ML detection path
    # ------------------------------------------------------------------

    def process_packet(
        self,
        packet: PacketRecord,
    ) -> dict[str, Any]:
        """Process one PacketRecord through N-BaIoT and ML prediction.

        Processing sequence:

            PacketRecord
                ↓
            N-BaIoTFeatureExtractor
                ↓
            GenericFeatureRecord
                ↓
            Predictor
                ↓
            PredictionResult
                ↓
            Database
                ↓
            AlertRecord if Attack
        """

        if not isinstance(packet, PacketRecord):
            raise TypeError(
                "packet must be a PacketRecord instance"
            )

        if self.predictor is None:
            raise ValueError(
                "predictor is required for packet detection"
            )

        if self.database is None:
            raise ValueError(
                "database is required for packet detection"
            )

        # --------------------------------------------------------------
        # STEP 1 — Extract 115 N-BaIoT features
        # --------------------------------------------------------------

        feature_record = (
            self.nbaiot_feature_extractor.extract_record(packet)
        )

        if not isinstance(feature_record, GenericFeatureRecord):
            raise TypeError(
                "N-BaIoT extractor must return a GenericFeatureRecord"
            )

        # --------------------------------------------------------------
        # STEP 2 — Run the configured ML predictor
        # --------------------------------------------------------------

        prediction = self.predictor.predict(feature_record)

        if not isinstance(prediction, PredictionResult):
            raise TypeError(
                "predictor must return a PredictionResult"
            )

        # --------------------------------------------------------------
        # STEP 3 — Initialize database
        # --------------------------------------------------------------

        self.database.initialize()

        # --------------------------------------------------------------
        # STEP 4 — Store feature record
        # --------------------------------------------------------------

        self.database.insert_feature_record(feature_record)

        # --------------------------------------------------------------
        # STEP 5 — Store prediction
        # --------------------------------------------------------------

        self.database.insert_prediction(prediction)

        # --------------------------------------------------------------
        # STEP 6 — Generate alert for an attack
        # --------------------------------------------------------------

        alert: AlertRecord | None = None

        if prediction.prediction.lower() == "attack":
            flow_id = str(
                feature_record.metadata.get(
                    "flow_id",
                    "unknown-flow",
                )
            )

            alert = AlertRecord(
                alert_id=(
                    f"alert-{flow_id}-"
                    f"{int(prediction.timestamp)}"
                ),
                flow_id=flow_id,
                timestamp=float(prediction.timestamp),
                prediction=prediction.prediction,
                probability=prediction.probability,
                status="new",
            )

            self.database.insert_alert(alert)

        # --------------------------------------------------------------
        # STEP 7 — Return complete processing result
        # --------------------------------------------------------------

        return {
            "packet": packet,
            "feature_record": feature_record,
            "prediction": prediction,
            "alert": alert,
            "persisted": True,
        }