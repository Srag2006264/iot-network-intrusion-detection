"""Detection pipeline for packet-level and flow-level intrusion detection."""

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
    """Connect packet/flow feature extraction, prediction, and persistence.

    The pipeline supports two detection paths:

    1. Flow-level:
       FlowRecord -> FlowFeatureExtractor -> Predictor

    2. Packet-level N-BaIoT:
       PacketRecord -> NBAIoTFeatureExtractor -> Predictor

    Both paths produce a standard PredictionResult and persist their
    results through the project's Database abstraction.
    """

    def __init__(
        self,
        feature_extractor: FlowFeatureExtractor | None = None,
        predictor: Predictor | None = None,
        database: Database | None = None,
        nbaiot_feature_extractor: NBAIoTFeatureExtractor | None = None,
    ) -> None:
        self.feature_extractor = feature_extractor or FlowFeatureExtractor()
        self.nbaiot_feature_extractor = (
            nbaiot_feature_extractor or NBAIoTFeatureExtractor()
        )
        self.predictor = predictor
        self.database = database

    # ------------------------------------------------------------------
    # Dependency validation
    # ------------------------------------------------------------------

    def _validate_dependencies(self) -> None:
        """Validate dependencies required for detection processing."""

        if self.predictor is None:
            raise ValueError("predictor is required for detection processing")

        if self.database is None:
            raise ValueError("database is required for detection processing")

    # ------------------------------------------------------------------
    # Alert handling
    # ------------------------------------------------------------------

    def _create_alert(
        self,
        flow_id: str,
        prediction: PredictionResult,
    ) -> AlertRecord | None:
        """Create an alert when the prediction represents an attack."""

        if prediction.prediction.lower() != "attack":
            return None

        return AlertRecord(
            alert_id=f"alert-{flow_id}-{int(prediction.timestamp)}",
            flow_id=flow_id,
            timestamp=float(prediction.timestamp),
            prediction=prediction.prediction,
            probability=prediction.probability,
            status="new",
        )

    def _persist_prediction(
        self,
        prediction: PredictionResult,
        alert: AlertRecord | None = None,
    ) -> None:
        """Persist a prediction and optional alert."""

        if self.database is None:
            raise ValueError("database is required for detection processing")

        self.database.insert_prediction(prediction)

        if alert is not None:
            self.database.insert_alert(alert)

    # ------------------------------------------------------------------
    # Existing flow-level pipeline
    # ------------------------------------------------------------------

    def process_flow(self, flow: FlowRecord) -> dict[str, Any]:
        """Process one flow through feature extraction, prediction, and persistence."""

        if not isinstance(flow, FlowRecord):
            raise TypeError("flow must be a FlowRecord instance")

        self._validate_dependencies()

        if self.database is None:
            raise ValueError("database is required for detection processing")

        # --------------------------------------------------------------
        # Flow-level feature extraction
        # --------------------------------------------------------------

        feature_record = self.feature_extractor.extract(flow)

        # --------------------------------------------------------------
        # Prediction
        # --------------------------------------------------------------

        prediction = self.predictor.predict(feature_record)

        if not isinstance(prediction, PredictionResult):
            raise TypeError("predictor must return a PredictionResult")

        # --------------------------------------------------------------
        # Persistence
        # --------------------------------------------------------------

        self.database.initialize()

        self.database.insert_flow(flow)
        self.database.insert_feature_record(feature_record)
        self.database.insert_prediction(prediction)

        # --------------------------------------------------------------
        # Alert
        # --------------------------------------------------------------

        alert = self._create_alert(
            flow_id=flow.flow_id,
            prediction=prediction,
        )

        if alert is not None:
            self.database.insert_alert(alert)

        return {
            "flow": flow,
            "feature_record": feature_record,
            "prediction": prediction,
            "alert": alert,
            "persisted": True,
        }

    # ------------------------------------------------------------------
    # Packet-level N-BaIoT pipeline
    # ------------------------------------------------------------------

    def process_packet(self, packet: PacketRecord) -> dict[str, Any]:
        """Process one packet through N-BaIoT, Random Forest, and persistence.

        Pipeline:

            PacketRecord
                -> NBAIoTFeatureExtractor
                -> GenericFeatureRecord
                -> Predictor
                -> PredictionResult
                -> Database
                -> optional AlertRecord
        """

        if not isinstance(packet, PacketRecord):
            raise TypeError("packet must be a PacketRecord instance")

        self._validate_dependencies()

        if self.database is None:
            raise ValueError("database is required for detection processing")

        # --------------------------------------------------------------
        # N-BaIoT feature extraction
        # --------------------------------------------------------------

        feature_record = self.nbaiot_feature_extractor.extract_record(packet)

        if not isinstance(feature_record, GenericFeatureRecord):
            raise TypeError(
                "N-BaIoT feature extractor must return a GenericFeatureRecord"
            )

        # --------------------------------------------------------------
        # Prediction
        # --------------------------------------------------------------

        prediction = self.predictor.predict(feature_record)

        if not isinstance(prediction, PredictionResult):
            raise TypeError("predictor must return a PredictionResult")

        # --------------------------------------------------------------
        # Determine flow identifier
        #
        # Packet-level detection does not have a FlowRecord yet.
        # The feature extractor stores a packet-derived identifier in
        # the GenericFeatureRecord metadata.
        # --------------------------------------------------------------

        flow_id = str(
            feature_record.metadata.get(
                "flow_id",
                f"packet-{packet.timestamp}",
            )
        )

        # --------------------------------------------------------------
        # Persistence
        # --------------------------------------------------------------

        self.database.initialize()

        self.database.insert_feature_record(feature_record)
        self.database.insert_prediction(prediction)

        # --------------------------------------------------------------
        # Alert
        # --------------------------------------------------------------

        alert = self._create_alert(
            flow_id=flow_id,
            prediction=prediction,
        )

        if alert is not None:
            self.database.insert_alert(alert)

        return {
            "packet": packet,
            "feature_record": feature_record,
            "prediction": prediction,
            "alert": alert,
            "persisted": True,
        }