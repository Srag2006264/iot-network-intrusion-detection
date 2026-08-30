"""Tests for grouping packets into bidirectional network flows."""

from scapy.all import IP, TCP, UDP, Ether

from src.core.contracts import PacketRecord
from src.network.flow import FlowBuilder


def _make_packet(src_ip: str, dst_ip: str, src_port: int, dst_port: int, protocol: str, timestamp: float, payload_len: int) -> PacketRecord:
    if protocol == "tcp":
        pkt = Ether() / IP(src=src_ip, dst=dst_ip) / TCP(sport=src_port, dport=dst_port)
    else:
        pkt = Ether() / IP(src=src_ip, dst=dst_ip) / UDP(sport=src_port, dport=dst_port)

    pkt.time = timestamp
    bytes_len = max(payload_len, len(pkt))
    pkt = Ether() / IP(src=src_ip, dst=dst_ip) / (TCP(sport=src_port, dport=dst_port) if protocol == "tcp" else UDP(sport=src_port, dport=dst_port))
    pkt.time = timestamp
    pkt = pkt.__class__(bytes(pkt))
    pkt.time = timestamp
    return PacketRecord(
        timestamp=timestamp,
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=src_port,
        dst_port=dst_port,
        protocol=protocol,
        packet_length=bytes_len,
        flags="S" if protocol == "tcp" else None,
    )


def test_packets_in_same_bi_directional_connection_are_grouped_into_one_flow():
    builder = FlowBuilder()

    packet_one = _make_packet("10.0.0.1", "10.0.0.2", 5000, 80, "tcp", 10.0, 100)
    packet_two = _make_packet("10.0.0.2", "10.0.0.1", 80, 5000, "tcp", 10.5, 120)

    builder.add_packet(packet_one)
    builder.add_packet(packet_two)

    flows = builder.get_flow_records()
    assert len(flows) == 1
    assert flows[0].packet_count == 2


def test_forward_and_reverse_packets_share_the_same_flow():
    builder = FlowBuilder()

    forward = _make_packet("192.168.1.2", "192.168.1.3", 5000, 80, "tcp", 1.0, 150)
    reverse = _make_packet("192.168.1.3", "192.168.1.2", 80, 5000, "tcp", 1.5, 200)

    builder.add_packet(forward)
    builder.add_packet(reverse)

    flows = builder.get_flow_records()
    assert len(flows) == 1
    assert flows[0].src_ip in {"192.168.1.2", "192.168.1.3"}
    assert flows[0].dst_ip in {"192.168.1.2", "192.168.1.3"}


def test_different_ports_or_protocols_create_different_flows():
    builder = FlowBuilder()

    builder.add_packet(_make_packet("10.0.0.1", "10.0.0.2", 5000, 80, "tcp", 2.0, 100))
    builder.add_packet(_make_packet("10.0.0.1", "10.0.0.2", 5001, 80, "tcp", 2.1, 100))
    builder.add_packet(_make_packet("10.0.0.1", "10.0.0.2", 5000, 80, "udp", 2.2, 100))

    assert builder.flow_count == 3


def test_flow_packet_count_is_correct():
    builder = FlowBuilder()

    for offset in range(3):
        builder.add_packet(_make_packet("10.0.0.1", "10.0.0.2", 5000, 80, "tcp", 3.0 + offset, 100))

    flows = builder.get_flow_records()
    assert flows[0].packet_count == 3


def test_flow_byte_count_is_correct():
    builder = FlowBuilder()

    builder.add_packet(_make_packet("10.0.0.1", "10.0.0.2", 5000, 80, "tcp", 4.0, 100))
    builder.add_packet(_make_packet("10.0.0.2", "10.0.0.1", 80, 5000, "tcp", 4.5, 150))

    flows = builder.get_flow_records()
    assert flows[0].byte_count == 250


def test_flow_start_and_end_timestamps_are_handled_correctly():
    builder = FlowBuilder()

    builder.add_packet(_make_packet("10.0.0.1", "10.0.0.2", 5000, 80, "tcp", 5.0, 100))
    builder.add_packet(_make_packet("10.0.0.2", "10.0.0.1", 80, 5000, "tcp", 7.0, 200))

    flows = builder.get_flow_records()
    assert flows[0].start_time == 5.0
    assert flows[0].end_time == 7.0
    assert flows[0].duration == 2.0


def test_empty_flow_state_behaves_correctly():
    builder = FlowBuilder()

    assert builder.flow_count == 0
    assert builder.get_flow_records() == []
    builder.clear()
    assert builder.flow_count == 0


def test_malformed_or_unsupported_packet_input_does_not_crash_flow_builder():
    builder = FlowBuilder()

    assert builder.add_packet(None) is None
    assert builder.add_packet("not-a-packet") is None
    assert builder.flow_count == 0
