"""Flow construction for normalized packet records."""

from __future__ import annotations

from typing import Any

from src.core.contracts import FlowRecord, PacketRecord


class FlowBuilder:
    """Combine packets into bidirectional connection flows.

    A flow key is based on the protocol and the two endpoints. Reverse traffic
    is mapped to the same flow so that traffic in both directions belongs to the
    same session.
    """

    def __init__(self) -> None:
        self._flows: dict[tuple[str, tuple[tuple[str, int | None], tuple[str, int | None]]], dict[str, Any]] = {}

    @staticmethod
    def _flow_key(packet: PacketRecord) -> tuple[str, tuple[tuple[str, int | None], tuple[str, int | None]]]:
        """Build a canonical bidirectional flow key for a packet."""
        first_endpoint = (packet.src_ip, packet.src_port)
        second_endpoint = (packet.dst_ip, packet.dst_port)
        endpoints = tuple(sorted((first_endpoint, second_endpoint), key=lambda item: (item[0], item[1] if item[1] is not None else -1)))
        return (packet.protocol, endpoints)

    @staticmethod
    def _flow_id_for_key(flow_key: tuple[str, tuple[tuple[str, int | None], tuple[str, int | None]]]) -> str:
        """Generate a deterministic flow identifier from a canonical key."""
        protocol, endpoints = flow_key
        values = [protocol]
        for endpoint in endpoints:
            values.append(endpoint[0])
            values.append(str(endpoint[1]))
        return "|".join(values)

    def add_packet(self, packet: PacketRecord | None) -> FlowRecord | None:
        """Add a PacketRecord to the flow state and return the current flow record."""
        if packet is None:
            return None
        if not isinstance(packet, PacketRecord):
            return None

        flow_key = self._flow_key(packet)
        if flow_key not in self._flows:
            self._flows[flow_key] = {
                "flow_id": self._flow_id_for_key(flow_key),
                "src_ip": packet.src_ip,
                "dst_ip": packet.dst_ip,
                "src_port": packet.src_port,
                "dst_port": packet.dst_port,
                "protocol": packet.protocol,
                "start_time": packet.timestamp,
                "end_time": packet.timestamp,
                "packet_count": 0,
                "byte_count": 0,
            }

        flow = self._flows[flow_key]
        flow["packet_count"] += 1
        flow["byte_count"] += packet.packet_length
        flow["start_time"] = min(flow["start_time"], packet.timestamp)
        flow["end_time"] = max(flow["end_time"], packet.timestamp)

        if flow["src_ip"] == packet.dst_ip and flow["dst_ip"] == packet.src_ip:
            # reverse packet, keep the original canonical direction as the first observed endpoint
            pass

        return self._build_flow_record(flow_key)

    def _build_flow_record(self, flow_key: tuple[str, tuple[tuple[str, int | None], tuple[str, int | None]]]) -> FlowRecord:
        """Create a FlowRecord from the current flow state."""
        flow = self._flows[flow_key]
        duration = max(0.0, float(flow["end_time"]) - float(flow["start_time"]))
        return FlowRecord(
            flow_id=flow["flow_id"],
            src_ip=flow["src_ip"],
            dst_ip=flow["dst_ip"],
            src_port=flow["src_port"],
            dst_port=flow["dst_port"],
            protocol=flow["protocol"],
            start_time=float(flow["start_time"]),
            end_time=float(flow["end_time"]),
            duration=duration,
            packet_count=int(flow["packet_count"]),
            byte_count=int(flow["byte_count"]),
            summary_stats={
                "packet_count": int(flow["packet_count"]),
                "byte_count": int(flow["byte_count"]),
                "duration": duration,
            },
        )

    def get_flow_records(self) -> list[FlowRecord]:
        """Return the current list of flow records."""
        return [self._build_flow_record(key) for key in self._flows]

    @property
    def flow_count(self) -> int:
        """Return the number of active flows."""
        return len(self._flows)

    def clear(self) -> None:
        """Clear all stored flows."""
        self._flows.clear()
