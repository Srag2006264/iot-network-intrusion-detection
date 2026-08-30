"""Packet capture abstraction built on top of Scapy.

This module is intentionally independent from the machine learning layer,
SQLite, and the dashboard. It converts supported packets into the shared
PacketRecord contract used by the rest of the project.
"""

from __future__ import annotations

from typing import Any

from scapy.all import IP, TCP, UDP, sniff

from src.core.contracts import PacketRecord


class PacketCapture:
    """Capture packets from a local interface or process packets supplied manually.

    This component is designed for authorized/local traffic only and must never
    be treated as a general offensive networking tool.
    """

    def __init__(
        self,
        interface: str | None = None,
        filter_expr: str | None = None,
    ) -> None:
        self.interface = interface
        self.filter_expr = filter_expr
        self._stopped = False
        self._captured_packets: list[PacketRecord] = []

    def process_packet(self, packet: Any) -> PacketRecord | None:
        """Convert a packet-like object into a PacketRecord when supported."""
        if packet is None:
            return None

        try:
            if not hasattr(packet, "time"):
                return None

            ip_layer = packet.getlayer(IP)
            if ip_layer is None:
                return None

            transport = packet.getlayer(TCP)
            if transport is None:
                transport = packet.getlayer(UDP)

            src_port = None
            dst_port = None
            protocol = "ip"

            if transport is not None:
                src_port = int(getattr(transport, "sport", 0) or 0)
                dst_port = int(getattr(transport, "dport", 0) or 0)
                protocol = "tcp" if isinstance(transport, TCP) else "udp"

            flags = None
            if isinstance(transport, TCP):
                flags = transport.flags

            return PacketRecord(
                timestamp=float(packet.time),
                src_ip=str(ip_layer.src),
                dst_ip=str(ip_layer.dst),
                src_port=src_port or None,
                dst_port=dst_port or None,
                protocol=protocol,
                packet_length=int(len(packet)),
                flags=str(flags) if flags is not None else None,
            )
        except Exception:
            return None

    def handle_packet(self, packet: Any) -> PacketRecord | None:
        """Process a packet and keep it in the in-memory capture list."""
        record = self.process_packet(packet)
        if record is not None:
            self._captured_packets.append(record)
        return record

    def start(self, count: int | None = None, timeout: float | None = None) -> list[PacketRecord]:
        """Start a live capture session using Scapy.

        This method is intended for authorized/local packet capture. It returns a
        list of PacketRecord values produced during the captured run.
        """
        self._stopped = False
        self._captured_packets = []
        collected: list[PacketRecord] = []

        def _callback(pkt: Any) -> None:
            if self._stopped:
                return
            record = self.process_packet(pkt)
            if record is not None:
                collected.append(record)
                self._captured_packets.append(record)

        sniff(
            iface=self.interface,
            filter=self.filter_expr,
            prn=_callback,
            store=False,
            count=count,
            timeout=timeout,
            stop_filter=lambda pkt: self._stopped,
        )

        return collected

    def stop(self) -> None:
        """Stop the current capture loop safely."""
        self._stopped = True

    def get_captured_packets(self) -> list[PacketRecord]:
        """Return a copy of the captured packets stored in memory."""
        return list(self._captured_packets)
