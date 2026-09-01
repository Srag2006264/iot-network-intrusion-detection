"""Streamlit dashboard for the IoT Network Intrusion Detection System.

The dashboard reads detection results from the project's SQLite database and
displays security monitoring information.

Architecture:

    Live Network Traffic
            |
            v
       N-BaIoT / Kitsune
            |
            v
        115 Features
            |
            v
       Random Forest
            |
            +------------------+
            |                  |
         Normal              Attack
            |                  |
            |                  v
            |             AlertRecord
            |                  |
            +--------+---------+
                     |
                     v
                   SQLite
                     |
                     v
                Streamlit UI
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from src.core.contracts import AlertRecord, FlowRecord, PredictionResult
from src.storage.database import Database


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="IoT IDS Dashboard",
    page_icon="🛡️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Dashboard class
# ---------------------------------------------------------------------------


class SecurityDashboard:
    """Read-only dashboard for IDS monitoring."""

    def __init__(
        self,
        database_path: str | Path | None = None,
    ) -> None:
        """Initialize the dashboard.

        If no database path is supplied, use the project's runtime database.
        """

        if database_path is not None:
            self.database_path = str(database_path)
        else:
            self.database_path = "data/ids.sqlite3"

        self.database = Database(self.database_path)

    # -----------------------------------------------------------------------
    # Database
    # -----------------------------------------------------------------------

    def _ensure_database_exists(self) -> None:
        """Initialize the database if required tables do not exist."""

        self.database.initialize()

    # -----------------------------------------------------------------------
    # Data access
    # -----------------------------------------------------------------------

    def get_flow_records(self) -> list[FlowRecord]:
        """Return all stored flow records."""

        self._ensure_database_exists()
        return self.database.get_all_flows()

    def get_predictions(self) -> list[PredictionResult]:
        """Return all stored prediction records."""

        self._ensure_database_exists()
        return self.database.get_all_predictions()

    def get_alerts(self) -> list[AlertRecord]:
        """Return all stored security alerts."""

        self._ensure_database_exists()
        return self.database.get_all_alerts()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    def get_summary(self) -> dict[str, Any]:
        """Calculate dashboard summary metrics."""

        flows = self.get_flow_records()
        predictions = self.get_predictions()
        alerts = self.get_alerts()

        normal_count = sum(
            1
            for prediction in predictions
            if prediction.prediction.lower() == "normal"
        )

        attack_count = sum(
            1
            for prediction in predictions
            if prediction.prediction.lower() == "attack"
        )

        total_packets = len(predictions)

        total_bytes = sum(
            flow.byte_count
            for flow in flows
        )

        return {
            "flow_count": len(flows),
            "packet_count": total_packets,
            "byte_count": total_bytes,
            "prediction_count": len(predictions),
            "normal_count": normal_count,
            "attack_count": attack_count,
            "alert_count": len(alerts),
            "flow_records": flows,
            "predictions": predictions,
            "alerts": alerts,
        }

    # -----------------------------------------------------------------------
    # Formatting helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def format_timestamp(timestamp: float) -> str:
        """Convert Unix timestamp into readable local time."""

        try:
            return datetime.fromtimestamp(timestamp).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        except (TypeError, ValueError, OSError):
            return str(timestamp)

    @staticmethod
    def format_probability(
        probability: float | None,
    ) -> str:
        """Convert probability to percentage text."""

        if probability is None:
            return "N/A"

        return f"{probability * 100:.2f}%"

    @staticmethod
    def get_confidence_assessment(
        prediction: PredictionResult,
    ) -> str:
        """Return a human-readable interpretation of model confidence.

        This does NOT change the Random Forest prediction.

        It only provides an additional dashboard interpretation.
        """

        probability = prediction.probability

        if probability is None:
            return "Unknown"

        prediction_name = prediction.prediction.lower()

        if prediction_name == "normal":
            if probability >= 0.80:
                return "High-confidence Normal"

            return "Normal / Borderline"

        if prediction_name == "attack":
            if probability >= 0.80:
                return "High-confidence Attack"

            return "Borderline Attack"

        return "Unknown"

    @staticmethod
    def get_confidence_level(
        prediction: PredictionResult,
    ) -> str:
        """Return a compact confidence category."""

        probability = prediction.probability

        if probability is None:
            return "Unknown"

        if probability >= 0.80:
            return "High"

        if probability >= 0.50:
            return "Medium"

        return "Low"

    # -----------------------------------------------------------------------
    # Sidebar
    # -----------------------------------------------------------------------

    def render_sidebar(self) -> None:
        """Render sidebar information and controls."""

        with st.sidebar:
            st.header("Database")

            st.code(
                self.database_path,
                language="text",
            )

            database_path = Path(self.database_path)

            if database_path.exists():
                st.success("Database connected")
            else:
                st.warning("Database will be created")

            if st.button(
                "🔄 Refresh data",
                width="stretch",
            ):
                st.rerun()

            st.divider()

            st.caption("Detection Configuration")

            st.write("**Feature Schema**")
            st.write("N-BaIoT / Kitsune")

            st.write("**Feature Count**")
            st.write("115")

            st.write("**Classifier**")
            st.write("Random Forest")

            st.divider()

            st.caption("Confidence Interpretation")

            st.write("**≥ 80%**")
            st.write("High confidence")

            st.write("**50% – < 80%**")
            st.write("Medium / borderline")

            st.write("**< 50%**")
            st.write("Low confidence")

    # -----------------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------------

    def render_header(self) -> None:
        """Render dashboard header."""

        st.title("🛡️ IoT Network Intrusion Detection System")

        st.caption(
            "Real-time security monitoring powered by "
            "N-BaIoT and Random Forest"
        )

    # -----------------------------------------------------------------------
    # Overview
    # -----------------------------------------------------------------------

    def render_overview(
        self,
        summary: dict[str, Any],
    ) -> None:
        """Render overview metrics."""

        st.header("📊 Detection Overview")

        col1, col2, col3, col4, col5, col6 = st.columns(6)

        col1.metric(
            "Packets",
            summary["packet_count"],
        )

        col2.metric(
            "Predictions",
            summary["prediction_count"],
        )

        col3.metric(
            "Normal",
            summary["normal_count"],
        )

        col4.metric(
            "Attacks",
            summary["attack_count"],
        )

        col5.metric(
            "Alerts",
            summary["alert_count"],
        )

        col6.metric(
            "Flows",
            summary["flow_count"],
        )

        # -------------------------------------------------------------------
        # Security status
        # -------------------------------------------------------------------

        if summary["alert_count"] > 0:
            st.error(
                f"🚨 {summary['alert_count']} security alert(s) "
                f"recorded."
            )
        elif summary["attack_count"] > 0:
            st.warning(
                f"⚠️ {summary['attack_count']} attack prediction(s) "
                f"detected."
            )
        else:
            st.success(
                "🟢 No attacks detected in recorded traffic."
            )

    # -----------------------------------------------------------------------
    # Prediction distribution
    # -----------------------------------------------------------------------

    def render_prediction_distribution(
        self,
        summary: dict[str, Any],
    ) -> None:
        """Render prediction distribution chart."""

        st.header("📈 Prediction Distribution")

        attack_count = summary["attack_count"]
        normal_count = summary["normal_count"]

        chart_data = {
            "Prediction": [
                "Attack",
                "Normal",
            ],
            "Count": [
                attack_count,
                normal_count,
            ],
        }

        st.bar_chart(
            chart_data,
            x="Prediction",
            y="Count",
            width="stretch",
        )

    # -----------------------------------------------------------------------
    # Recent predictions
    # -----------------------------------------------------------------------

    def render_predictions(
        self,
        summary: dict[str, Any],
    ) -> None:
        """Render recent prediction records."""

        st.header("🔎 Recent Predictions")

        predictions = summary["predictions"]

        if not predictions:
            st.info("No prediction records available.")
            return

        recent_predictions = list(
            reversed(predictions[-10:])
        )

        prediction_rows = []

        for prediction in recent_predictions:
            prediction_rows.append(
                {
                    "Time": self.format_timestamp(
                        prediction.timestamp
                    ),
                    "Flow / Packet ID": prediction.flow_id,
                    "Prediction": prediction.prediction,
                    "Confidence": self.format_probability(
                        prediction.probability
                    ),
                    "Confidence Level": self.get_confidence_level(
                        prediction
                    ),
                    "Assessment": self.get_confidence_assessment(
                        prediction
                    ),
                    "Model": prediction.model_name,
                    "Source": prediction.source,
                }
            )

        st.dataframe(
            prediction_rows,
            width="stretch",
            hide_index=True,
        )

    # -----------------------------------------------------------------------
    # Security alerts
    # -----------------------------------------------------------------------

    def render_alerts(
        self,
        summary: dict[str, Any],
    ) -> None:
        """Render security alerts."""

        st.header("🚨 Security Alerts")

        alerts = summary["alerts"]

        if not alerts:
            st.success(
                "No security alerts have been generated."
            )
            return

        recent_alerts = list(
            reversed(alerts[-10:])
        )

        for alert in recent_alerts:
            probability = self.format_probability(
                alert.probability
            )

            # ---------------------------------------------------------------
            # Determine alert confidence category
            # ---------------------------------------------------------------

            if alert.probability is None:
                confidence_level = "Unknown"
            elif alert.probability >= 0.80:
                confidence_level = "High"
            elif alert.probability >= 0.50:
                confidence_level = "Medium / Borderline"
            else:
                confidence_level = "Low"

            if confidence_level == "High":
                st.error(
                    f"🚨 ATTACK DETECTED | "
                    f"Confidence: {probability} | "
                    f"Severity: High | "
                    f"Status: {alert.status}"
                )

            else:
                st.warning(
                    f"⚠️ ATTACK PREDICTION | "
                    f"Confidence: {probability} | "
                    f"Severity: {confidence_level} | "
                    f"Status: {alert.status}"
                )

            alert_rows = [
                {
                    "Alert ID": alert.alert_id,
                    "Flow / Packet ID": alert.flow_id,
                    "Prediction": alert.prediction,
                    "Confidence": probability,
                    "Confidence Level": confidence_level,
                    "Time": self.format_timestamp(
                        alert.timestamp
                    ),
                    "Status": alert.status,
                }
            ]

            st.dataframe(
                alert_rows,
                width="stretch",
                hide_index=True,
            )

    # -----------------------------------------------------------------------
    # Network flows
    # -----------------------------------------------------------------------

    def render_flows(
        self,
        summary: dict[str, Any],
    ) -> None:
        """Render stored flow records."""

        st.header("🌐 Recent Network Flows")

        flows = summary["flow_records"]

        if not flows:
            st.info(
                "No FlowRecord entries available. "
                "The N-BaIoT live detector currently operates "
                "at packet level."
            )
            return

        recent_flows = list(
            reversed(flows[-10:])
        )

        flow_rows = []

        for flow in recent_flows:
            flow_rows.append(
                {
                    "Flow ID": flow.flow_id,
                    "Source": f"{flow.src_ip}:{flow.src_port}",
                    "Destination": f"{flow.dst_ip}:{flow.dst_port}",
                    "Protocol": flow.protocol,
                    "Packets": flow.packet_count,
                    "Bytes": flow.byte_count,
                    "Duration": f"{flow.duration:.6f}s",
                    "Start": self.format_timestamp(
                        flow.start_time
                    ),
                    "End": self.format_timestamp(
                        flow.end_time
                    ),
                }
            )

        st.dataframe(
            flow_rows,
            width="stretch",
            hide_index=True,
        )

    # -----------------------------------------------------------------------
    # Detection system
    # -----------------------------------------------------------------------

    def render_detection_system(self) -> None:
        """Render IDS architecture information."""

        st.header("⚙️ Detection System")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Feature Schema",
            "N-BaIoT / Kitsune",
        )

        col2.metric(
            "Features",
            "115",
        )

        col3.metric(
            "Classifier",
            "Random Forest",
        )

        st.caption(
            "Network packet → N-BaIoT/Kitsune features → "
            "Random Forest → Prediction → Alert → SQLite"
        )

        st.divider()

        st.subheader("Detection Confidence")

        st.write(
            """
            The Random Forest produces the primary binary prediction:
            **Normal** or **Attack**.

            The dashboard additionally interprets the model probability:

            - **≥ 80%** → High confidence
            - **50% – < 80%** → Medium / borderline confidence
            - **< 50%** → Low confidence

            This interpretation does not modify the trained model's
            prediction.
            """
        )

    # -----------------------------------------------------------------------
    # Main render
    # -----------------------------------------------------------------------

    def render(self) -> None:
        """Render the complete dashboard."""

        self.render_sidebar()

        self.render_header()

        summary = self.get_summary()

        # -------------------------------------------------------------------
        # Empty database
        # -------------------------------------------------------------------

        if (
            not summary["predictions"]
            and not summary["alerts"]
            and not summary["flow_records"]
        ):
            st.info(
                "No telemetry has been recorded yet. "
                "Start the live detector to capture network traffic."
            )

            self.render_detection_system()

            return

        # -------------------------------------------------------------------
        # Dashboard sections
        # -------------------------------------------------------------------

        self.render_overview(summary)

        self.render_prediction_distribution(summary)

        self.render_predictions(summary)

        self.render_alerts(summary)

        self.render_flows(summary)

        self.render_detection_system()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Application entry point."""

    dashboard = SecurityDashboard()

    dashboard.render()


if __name__ == "__main__":
    main()