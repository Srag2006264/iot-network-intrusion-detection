"""Shared data contracts used across the intrusion detection pipeline.

These contracts are intentionally generic and model-agnostic. They describe
network-level objects and prediction outputs without assuming the final
N-BaIoT schema or a specific machine learning implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PacketRecord:
    """Normalized packet-level information captured from the network."""

    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int | None = None
    dst_port: int | None = None
    protocol: str = "ip"
    packet_length: int = 0
    flags: str | None = None


@dataclass(slots=True)
class FlowRecord:
    """Aggregated communication summary for a network flow."""

    flow_id: str
    src_ip: str
    dst_ip: str
    src_port: int | None = None
    dst_port: int | None = None
    protocol: str = "ip"
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    packet_count: int = 0
    byte_count: int = 0
    summary_stats: dict[str, float | int] = field(default_factory=dict)


@dataclass(slots=True)
class GenericFeatureRecord:
    """Generic network statistics extracted from a flow.

    This record intentionally remains generic and does not claim to represent
    the final N-BaIoT feature schema. The exact model-specific schema is
    handled later in the prediction layer.
    """

    feature_version: str
    features: dict[str, float | int | bool]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PredictionResult:
    """Standard prediction output produced by any predictor implementation."""

    flow_id: str
    prediction: str
    probability: float | None
    timestamp: float
    model_name: str
    source: str
    notes: str | None = None


@dataclass(slots=True)
class AlertRecord:
    """Database-friendly alert record for persistence and display."""

    alert_id: str
    flow_id: str
    timestamp: float
    prediction: str
    probability: float | None
    status: str = "new"
