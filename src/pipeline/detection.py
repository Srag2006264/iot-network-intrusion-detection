"""Detection pipeline for generic feature extraction and prediction."""

from __future__ import annotations

from typing import Any

from src.core.contracts import AlertRecord, FlowRecord, GenericFeatureRecord, PredictionResult
from src.network.features import FlowFeatureExtractor
from src.prediction.predictor import Predictor
from src.storage.database import Database


class DetectionPipeline:
    """Connect flow extraction, prediction, and persistence.

    This service intentionally depends only on abstractions already defined by
    the project: the feature extractor, the Predictor interface, and the SQLite
    database wrapper. It does not couple itself to Scapy, the Random Forest, or
    the dashboard.
    """

    def __init__(
        self,
        feature_extractor: FlowFeatureExtractor | None = None,
        predictor: Predictor | None = None,
        database: Database | None = None,
    ) -> None:
        self.feature_extractor = feature_extractor or FlowFeatureExtractor()
        self.predictor = predictor
        self.database = database

    def process_flow(self, flow: FlowRecord) -> dict[str, Any]:
        """Process one flow through feature extraction, prediction, and persistence."""
        if not isinstance(flow, FlowRecord):
            raise TypeError("flow must be a FlowRecord instance")
        if self.predictor is None:
            raise ValueError("predictor is required for detection processing")
        if self.database is None:
            raise ValueError("database is required for detection processing")

        feature_record = self.feature_extractor.extract(flow)
        prediction = self.predictor.predict(feature_record)

        if not isinstance(prediction, PredictionResult):
            raise TypeError("predictor must return a PredictionResult")

        self.database.initialize()
        self.database.insert_flow(flow)
        self.database.insert_feature_record(feature_record)
        self.database.insert_prediction(prediction)

        alert: AlertRecord | None = None
        if prediction.prediction.lower() == "attack":
            alert = AlertRecord(
                alert_id=f"alert-{flow.flow_id}-{int(prediction.timestamp)}",
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
