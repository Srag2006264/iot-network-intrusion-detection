"""Tests for packet capture conversion behavior."""

from scapy.all import IP, TCP, UDP, Ether

from src.core.contracts import PacketRecord
from src.network.capture import PacketCapture


def test_valid_packet_can_be_converted_to_packet_record():
    pkt = Ether() / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=1234, dport=80)
    pkt.time = 10.0

    capture = PacketCapture()
    record = capture.process_packet(pkt)

    assert isinstance(record, PacketRecord)
    assert record.src_ip == "10.0.0.1"
    assert record.dst_ip == "10.0.0.2"
    assert record.src_port == 1234
    assert record.dst_port == 80
    assert record.protocol == "tcp"
    assert record.packet_length > 0


def test_unsupported_packets_are_handled_safely():
    capture = PacketCapture()

    assert capture.process_packet(None) is None
    assert capture.process_packet(object()) is None


def test_packet_capture_accepts_manual_packet_processing():
    pkt = Ether() / IP(src="192.168.1.10", dst="192.168.1.20") / UDP(sport=5000, dport=53)
    pkt.time = 42.0

    capture = PacketCapture()
    record = capture.handle_packet(pkt)

    assert isinstance(record, PacketRecord)
    assert record.protocol == "udp"
    assert len(capture.get_captured_packets()) == 1


def test_malformed_packet_does_not_crash_capture_processing():
    capture = PacketCapture()

    malformed = object()
    assert capture.process_packet(malformed) is None
    assert capture.get_captured_packets() == []
