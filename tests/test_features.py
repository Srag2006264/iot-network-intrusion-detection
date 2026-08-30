"""Tests for generic flow feature extraction."""

from src.core.contracts import FlowRecord, GenericFeatureRecord
from src.network.features import FlowFeatureExtractor


def test_normal_flow_feature_extraction():
    flow = FlowRecord(
        flow_id="flow-001",
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        src_port=5000,
        dst_port=80,
        protocol="tcp",
        start_time=1.0,
        end_time=3.0,
        duration=2.0,
        packet_count=4,
        byte_count=200,
        summary_stats={"min_packet_size": 20, "max_packet_size": 80},
    )

    result = FlowFeatureExtractor().extract(flow)

    assert isinstance(result, GenericFeatureRecord)
    assert result.feature_version == "generic-flow-v1"
    assert result.features["packet_count"] == 4
    assert result.features["byte_count"] == 200
    assert result.features["duration"] == 2.0
    assert result.features["average_packet_size"] == 50.0
    assert result.features["minimum_packet_size"] == 20
    assert result.features["maximum_packet_size"] == 80
    assert result.features["packet_rate"] == 2.0
    assert result.features["byte_rate"] == 100.0


def test_packet_count_is_preserved():
    flow = FlowRecord(
        flow_id="flow-002",
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        protocol="udp",
        packet_count=7,
        byte_count=350,
        duration=5.0,
    )

    result = FlowFeatureExtractor().extract(flow)
    assert result.features["packet_count"] == 7


def test_byte_count_is_preserved():
    flow = FlowRecord(
        flow_id="flow-003",
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        protocol="tcp",
        packet_count=3,
        byte_count=240,
        duration=4.0,
    )

    result = FlowFeatureExtractor().extract(flow)
    assert result.features["byte_count"] == 240


def test_duration_is_used_in_rates():
    flow = FlowRecord(
        flow_id="flow-004",
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        protocol="tcp",
        packet_count=10,
        byte_count=400,
        duration=2.5,
    )

    result = FlowFeatureExtractor().extract(flow)
    assert result.features["duration"] == 2.5
    assert result.features["packet_rate"] == 4.0
    assert result.features["byte_rate"] == 160.0


def test_average_packet_size_is_calculated():
    flow = FlowRecord(
        flow_id="flow-005",
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        protocol="tcp",
        packet_count=5,
        byte_count=125,
        duration=1.0,
    )

    result = FlowFeatureExtractor().extract(flow)
    assert result.features["average_packet_size"] == 25.0


def test_minimum_and_maximum_packet_size_are_supported_when_present():
    flow = FlowRecord(
        flow_id="flow-006",
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        protocol="tcp",
        packet_count=4,
        byte_count=100,
        duration=1.0,
        summary_stats={"min_packet_size": 10, "max_packet_size": 50},
    )

    result = FlowFeatureExtractor().extract(flow)
    assert result.features["minimum_packet_size"] == 10
    assert result.features["maximum_packet_size"] == 50


def test_zero_duration_flow_has_safe_rates():
    flow = FlowRecord(
        flow_id="flow-007",
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        protocol="tcp",
        packet_count=3,
        byte_count=90,
        duration=0.0,
    )

    result = FlowFeatureExtractor().extract(flow)
    assert result.features["packet_rate"] == 0.0
    assert result.features["byte_rate"] == 0.0


def test_single_packet_flow_is_handled_without_division_error():
    flow = FlowRecord(
        flow_id="flow-008",
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        protocol="udp",
        packet_count=1,
        byte_count=60,
        duration=2.0,
    )

    result = FlowFeatureExtractor().extract(flow)
    assert result.features["packet_count"] == 1
    assert result.features["average_packet_size"] == 60.0
    assert result.features["packet_rate"] == 0.5
    assert result.features["byte_rate"] == 30.0


def test_empty_or_zero_byte_flow_is_safe():
    flow = FlowRecord(
        flow_id="flow-009",
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        protocol="icmp",
        packet_count=0,
        byte_count=0,
        duration=0.0,
    )

    result = FlowFeatureExtractor().extract(flow)
    assert result.features["packet_count"] == 0
    assert result.features["byte_count"] == 0
    assert result.features["average_packet_size"] == 0.0
    assert result.features["minimum_packet_size"] == 0.0
    assert result.features["maximum_packet_size"] == 0.0


def test_invalid_numeric_values_do_not_crash_extraction():
    flow = FlowRecord(
        flow_id="flow-010",
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        protocol="tcp",
        packet_count=-2,
        byte_count=-10,
        duration=-1.0,
        summary_stats={"min_packet_size": "bad", "max_packet_size": "x"},
    )

    result = FlowFeatureExtractor().extract(flow)
    assert result.features["packet_count"] == 0
    assert result.features["byte_count"] == 0
    assert result.features["duration"] == 0.0
    assert result.features["average_packet_size"] == 0.0


def test_repeated_extraction_is_deterministic():
    flow = FlowRecord(
        flow_id="flow-011",
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        protocol="tcp",
        packet_count=6,
        byte_count=300,
        duration=3.0,
        summary_stats={"min_packet_size": 30, "max_packet_size": 70},
    )

    first = FlowFeatureExtractor().extract(flow)
    second = FlowFeatureExtractor().extract(flow)

    assert first == second


def test_non_flow_input_raises_type_error():
    extractor = FlowFeatureExtractor()

    try:
        extractor.extract(object())
        assert False, "Expected TypeError for invalid flow object"
    except TypeError:
        pass
