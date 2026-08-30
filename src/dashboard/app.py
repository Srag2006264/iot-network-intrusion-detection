"""Streamlit dashboard for security monitoring.

This module reads persisted project records from SQLite and renders a simple
security monitoring dashboard. It does not capture traffic, run model
inference, or manipulate the database schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from src.core.contracts import AlertRecord, FlowRecord, PredictionResult
from src.storage.database import Database


class SecurityDashboard:
    """Read-only dashboard data access and presentation layer."""

    def __init__(self, database_path: str | Path | None = None) -> None:
        self.database_path = str(database_path) if database_path is not None else "iot_security.db"
        self.database = Database(self.database_path)

    def _ensure_database_exists(self) -> None:
        """Initialize the database if it has not been created yet."""
        self.database.initialize()

    def get_flow_records(self) -> list[FlowRecord]:
        """Return all stored flow records."""
        self._ensure_database_exists()
        return self.database.get_all_flows()

    def get_predictions(self) -> list[PredictionResult]:
        """Return all stored predictions."""
        self._ensure_database_exists()
        return self.database.get_all_predictions()

    def get_alerts(self) -> list[AlertRecord]:
        """Return all stored alerts."""
        self._ensure_database_exists()
        return self.database.get_all_alerts()

    def get_summary(self) -> dict[str, Any]:
        """Return summary metrics for the dashboard."""
        flows = self.get_flow_records()
        predictions = self.get_predictions()
        alerts = self.get_alerts()

        total_packets = sum(flow.packet_count for flow in flows)
        total_bytes = sum(flow.byte_count for flow in flows)
        normal_count = sum(1 for prediction in predictions if prediction.prediction.lower() == "normal")
        attack_count = sum(1 for prediction in predictions if prediction.prediction.lower() == "attack")

        return {
            "flow_count": len(flows),
            "packet_count": total_packets,
            "byte_count": total_bytes,
            "normal_count": normal_count,
            "attack_count": attack_count,
            "flow_records": flows,
            "predictions": predictions,
            "alerts": alerts,
        }

    def render(self) -> None:
        """Render the dashboard user interface."""
        st.title("IoT Network Security Monitoring")

        summary = self.get_summary()

        if not summary["flow_records"] and not summary["predictions"] and not summary["alerts"]:
            st.info("No telemetry has been recorded yet. The database is empty.")
            return

        st.subheader("Overview")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Flows", summary["flow_count"])
        col2.metric("Packets", summary["packet_count"])
        col3.metric("Bytes", summary["byte_count"])
        col4.metric("Normal", summary["normal_count"])
        col5.metric("Attack", summary["attack_count"])

        st.subheader("Recent predictions")
        if summary["predictions"]:
            prediction_df = [
                {
                    "flow_id": item.flow_id,
                    "prediction": item.prediction,
                    "probability": item.probability,
                    "timestamp": item.timestamp,
                    "model_name": item.model_name,
                    "source": item.source,
                }
                for item in summary["predictions"][-10:]
            ]
            st.dataframe(prediction_df, use_container_width=True)
        else:
            st.write("No prediction records available.")

        st.subheader("Recent alerts")
        if summary["alerts"]:
            alert_df = [
                {
                    "alert_id": item.alert_id,
                    "flow_id": item.flow_id,
                    "prediction": item.prediction,
                    "probability": item.probability,
                    "timestamp": item.timestamp,
                    "status": item.status,
                }
                for item in summary["alerts"][-10:]
            ]
            st.dataframe(alert_df, use_container_width=True)
        else:
            st.write("No alerts recorded.")

        st.subheader("Recent flows")
        if summary["flow_records"]:
            flow_df = [
                {
                    "flow_id": item.flow_id,
                    "src_ip": item.src_ip,
                    "dst_ip": item.dst_ip,
                    "protocol": item.protocol,
                    "packet_count": item.packet_count,
                    "byte_count": item.byte_count,
                    "duration": item.duration,
                    "start_time": item.start_time,
                    "end_time": item.end_time,
                }
                for item in summary["flow_records"][-10:]
            ]
            st.dataframe(flow_df, use_container_width=True)
        else:
            st.write("No flow records available.")


def main() -> None:
    """Entry point for the dashboard application."""
    dashboard = SecurityDashboard()
    dashboard.render()


if __name__ == "__main__":
    main()
