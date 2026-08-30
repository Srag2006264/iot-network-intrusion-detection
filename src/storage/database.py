"""SQLite persistence layer for generic project records.

This module stores the project’s shared dataclasses without referencing Scapy,
Random Forest, or dashboard code. The database remains generic and stores only
values passed to it by the surrounding pipeline.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from src.core.contracts import AlertRecord, FlowRecord, GenericFeatureRecord, PredictionResult


class Database:
    """Small SQLite wrapper for project records."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._connection: sqlite3.Connection | None = None

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Open a SQLite connection with row factory for easier retrieval."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def initialize(self) -> None:
        """Create required tables if they do not already exist."""
        with self.connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS flows (
                    flow_id TEXT PRIMARY KEY,
                    src_ip TEXT NOT NULL,
                    dst_ip TEXT NOT NULL,
                    src_port INTEGER,
                    dst_port INTEGER,
                    protocol TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    duration REAL NOT NULL,
                    packet_count INTEGER NOT NULL,
                    byte_count INTEGER NOT NULL,
                    summary_stats TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    flow_id TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    probability REAL,
                    timestamp REAL NOT NULL,
                    model_name TEXT NOT NULL,
                    source TEXT NOT NULL,
                    notes TEXT,
                    PRIMARY KEY (flow_id, timestamp)
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    flow_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    prediction TEXT NOT NULL,
                    probability REAL,
                    status TEXT NOT NULL DEFAULT 'new'
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feature_records (
                    flow_id TEXT PRIMARY KEY,
                    feature_version TEXT NOT NULL,
                    features TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )

            conn.commit()

    def insert_flow(self, flow: FlowRecord) -> None:
        """Insert a FlowRecord into the database."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO flows (
                    flow_id, src_ip, dst_ip, src_port, dst_port, protocol,
                    start_time, end_time, duration, packet_count, byte_count,
                    summary_stats
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    flow.flow_id,
                    flow.src_ip,
                    flow.dst_ip,
                    flow.src_port,
                    flow.dst_port,
                    flow.protocol,
                    float(flow.start_time),
                    float(flow.end_time),
                    float(flow.duration),
                    int(flow.packet_count),
                    int(flow.byte_count),
                    json.dumps(flow.summary_stats, sort_keys=True),
                ),
            )
            conn.commit()

    def get_flow(self, flow_id: str) -> FlowRecord | None:
        """Retrieve a FlowRecord by flow_id."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM flows WHERE flow_id = ?",
                (flow_id,),
            ).fetchone()

        if row is None:
            return None

        return FlowRecord(
            flow_id=row["flow_id"],
            src_ip=row["src_ip"],
            dst_ip=row["dst_ip"],
            src_port=row["src_port"],
            dst_port=row["dst_port"],
            protocol=row["protocol"],
            start_time=float(row["start_time"]),
            end_time=float(row["end_time"]),
            duration=float(row["duration"]),
            packet_count=int(row["packet_count"]),
            byte_count=int(row["byte_count"]),
            summary_stats=json.loads(row["summary_stats"]),
        )

    def insert_feature_record(self, features: GenericFeatureRecord) -> None:
        """Insert a GenericFeatureRecord into the database."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO feature_records (
                    flow_id, feature_version, features, metadata
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    str(features.metadata.get("flow_id", "unknown")),
                    features.feature_version,
                    json.dumps(features.features, sort_keys=True),
                    json.dumps(features.metadata, sort_keys=True),
                ),
            )
            conn.commit()

    def get_feature_record(self, flow_id: str) -> GenericFeatureRecord | None:
        """Retrieve a GenericFeatureRecord associated with a flow."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM feature_records WHERE flow_id = ?",
                (flow_id,),
            ).fetchone()

        if row is None:
            return None

        return GenericFeatureRecord(
            feature_version=row["feature_version"],
            features=json.loads(row["features"]),
            metadata=json.loads(row["metadata"]),
        )

    def insert_prediction(self, prediction: PredictionResult) -> None:
        """Insert a PredictionResult into the database."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO predictions (
                    flow_id, prediction, probability, timestamp, model_name, source, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prediction.flow_id,
                    prediction.prediction,
                    prediction.probability,
                    float(prediction.timestamp),
                    prediction.model_name,
                    prediction.source,
                    prediction.notes,
                ),
            )
            conn.commit()

    def get_prediction(self, flow_id: str, timestamp: float | None = None) -> PredictionResult | None:
        """Retrieve a prediction result for a flow, optionally by timestamp."""
        with self.connection() as conn:
            if timestamp is None:
                row = conn.execute(
                    "SELECT * FROM predictions WHERE flow_id = ? ORDER BY timestamp DESC LIMIT 1",
                    (flow_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM predictions WHERE flow_id = ? AND timestamp = ?",
                    (flow_id, float(timestamp)),
                ).fetchone()

        if row is None:
            return None

        return PredictionResult(
            flow_id=row["flow_id"],
            prediction=row["prediction"],
            probability=row["probability"],
            timestamp=float(row["timestamp"]),
            model_name=row["model_name"],
            source=row["source"],
            notes=row["notes"],
        )

    def insert_alert(self, alert: AlertRecord) -> None:
        """Insert an AlertRecord into the database."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO alerts (
                    alert_id, flow_id, timestamp, prediction, probability, status
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    alert.alert_id,
                    alert.flow_id,
                    float(alert.timestamp),
                    alert.prediction,
                    alert.probability,
                    alert.status,
                ),
            )
            conn.commit()

    def get_alert(self, alert_id: str) -> AlertRecord | None:
        """Retrieve an alert by alert_id."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM alerts WHERE alert_id = ?",
                (alert_id,),
            ).fetchone()

        if row is None:
            return None

        return AlertRecord(
            alert_id=row["alert_id"],
            flow_id=row["flow_id"],
            timestamp=float(row["timestamp"]),
            prediction=row["prediction"],
            probability=row["probability"],
            status=row["status"],
        )

    def get_all_flows(self) -> list[FlowRecord]:
        """Return all stored flows."""
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM flows ORDER BY start_time ASC").fetchall()

        return [
            FlowRecord(
                flow_id=row["flow_id"],
                src_ip=row["src_ip"],
                dst_ip=row["dst_ip"],
                src_port=row["src_port"],
                dst_port=row["dst_port"],
                protocol=row["protocol"],
                start_time=float(row["start_time"]),
                end_time=float(row["end_time"]),
                duration=float(row["duration"]),
                packet_count=int(row["packet_count"]),
                byte_count=int(row["byte_count"]),
                summary_stats=json.loads(row["summary_stats"]),
            )
            for row in rows
        ]

    def get_all_predictions(self) -> list[PredictionResult]:
        """Return all stored predictions."""
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM predictions ORDER BY timestamp ASC").fetchall()

        return [
            PredictionResult(
                flow_id=row["flow_id"],
                prediction=row["prediction"],
                probability=row["probability"],
                timestamp=float(row["timestamp"]),
                model_name=row["model_name"],
                source=row["source"],
                notes=row["notes"],
            )
            for row in rows
        ]

    def get_all_alerts(self) -> list[AlertRecord]:
        """Return all stored alerts."""
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM alerts ORDER BY timestamp ASC").fetchall()

        return [
            AlertRecord(
                alert_id=row["alert_id"],
                flow_id=row["flow_id"],
                timestamp=float(row["timestamp"]),
                prediction=row["prediction"],
                probability=row["probability"],
                status=row["status"],
            )
            for row in rows
        ]

    def close(self) -> None:
        """Close the current connection if one exists."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
