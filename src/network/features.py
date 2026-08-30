"""Generic feature extraction for completed network flows.

This module converts a FlowRecord into a GenericFeatureRecord. The features are
kept intentionally generic and describe observable flow behavior only. They do
not claim to reproduce the final N-BaIoT feature schema or any model-specific
preprocessing.
"""

from __future__ import annotations

from typing import Any

from src.core.contracts import FlowRecord, GenericFeatureRecord


class FlowFeatureExtractor:
    """Produce generic, model-independent features from a flow record."""

    FEATURE_VERSION = "generic-flow-v1"

    @staticmethod
    def _coerce_number(value: Any, default: float = 0.0) -> float:
        """Convert a value to a float while guarding against invalid inputs."""
        if isinstance(value, bool):
            return float(default)
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def extract(self, flow: FlowRecord) -> GenericFeatureRecord:
        """Extract generic flow statistics from a FlowRecord."""
        if not isinstance(flow, FlowRecord):
            raise TypeError("flow must be a FlowRecord instance")

        packet_count = max(0, int(flow.packet_count))
        byte_count = max(0, int(flow.byte_count))
        duration = max(0.0, self._coerce_number(flow.duration, 0.0))

        average_packet_size = (byte_count / packet_count) if packet_count > 0 else 0.0

        summary_stats = flow.summary_stats or {}
        min_packet_size = self._coerce_number(summary_stats.get("min_packet_size"), average_packet_size)
        max_packet_size = self._coerce_number(summary_stats.get("max_packet_size"), average_packet_size)

        if packet_count == 0:
            min_packet_size = 0.0
            max_packet_size = 0.0
            average_packet_size = 0.0

        packet_rate = (packet_count / duration) if duration > 0.0 else 0.0
        byte_rate = (byte_count / duration) if duration > 0.0 else 0.0

        features = {
            "packet_count": packet_count,
            "byte_count": byte_count,
            "duration": duration,
            "average_packet_size": average_packet_size,
            "minimum_packet_size": min_packet_size,
            "maximum_packet_size": max_packet_size,
            "packet_rate": packet_rate,
            "byte_rate": byte_rate,
        }

        metadata = {
            "flow_id": flow.flow_id,
            "protocol": flow.protocol,
            "src_ip": flow.src_ip,
            "dst_ip": flow.dst_ip,
            "src_port": flow.src_port,
            "dst_port": flow.dst_port,
            "start_time": flow.start_time,
            "end_time": flow.end_time,
            "feature_source": "generic_flow_feature_extractor",
        }

        return GenericFeatureRecord(
            feature_version=self.FEATURE_VERSION,
            features=features,
            metadata=metadata,
        )


def extract_flow_features(flow: FlowRecord) -> GenericFeatureRecord:
    """Convenience function for flow feature extraction."""
    return FlowFeatureExtractor().extract(flow)
