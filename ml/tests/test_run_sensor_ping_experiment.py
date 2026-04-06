from __future__ import annotations

from ml.training.run_sensor_ping_experiment import build_netem_tokens, parse_ping_output_line
from ml.training.sensor_experiment_dataset import SensorExperimentPhase


def test_parse_ping_output_line_parses_reply() -> None:
    event = parse_ping_output_line(
        "[1776000000.123456] 64 bytes from 1.1.1.1: icmp_seq=7 ttl=57 time=23.4 ms"
    )

    assert event is not None
    assert event["event_type"] == "reply"
    assert event["icmp_seq"] == 7
    assert event["packets_lost"] == 0
    assert event["rtt_ms"] == 23.4


def test_parse_ping_output_line_parses_timeout() -> None:
    event = parse_ping_output_line("[1776000001.000001] no answer yet for icmp_seq=9")

    assert event is not None
    assert event["event_type"] == "timeout"
    assert event["icmp_seq"] == 9
    assert event["packets_lost"] == 1
    assert event["rtt_ms"] == 0.0


def test_build_netem_tokens_renders_delay_loss_rate() -> None:
    phase = SensorExperimentPhase(
        name="impairment",
        start_offset_seconds=0.0,
        duration_seconds=10.0,
        label=1,
        impairment_type="netem",
        severity="moderate",
        netem={
            "delay_ms": 120,
            "jitter_ms": 20,
            "loss_percent": 1.5,
            "rate_kbit": 2048,
        },
    )

    assert build_netem_tokens(phase) == [
        "delay",
        "120ms",
        "20ms",
        "loss",
        "1.5%",
        "rate",
        "2048kbit",
    ]
