"""Core project contracts and shared data models."""

from .contracts import (
    AlertRecord,
    FlowRecord,
    GenericFeatureRecord,
    PacketRecord,
    PredictionResult,
)

__all__ = [
    "PacketRecord",
    "FlowRecord",
    "GenericFeatureRecord",
    "PredictionResult",
    "AlertRecord",
]
