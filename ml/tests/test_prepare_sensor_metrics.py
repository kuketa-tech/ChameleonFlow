from __future__ import annotations

import pandas as pd

from ml.training.prepare_sensor_metrics import build_sensor_metrics_frame


def test_build_sensor_metrics_frame_normalizes_alias_columns() -> None:
    raw_frame = pd.DataFrame(
        {
            "flow_id": ["f1", "f1"],
            "time": ["2026-04-06T10:00:00Z", "2026-04-06T10:00:01Z"],
            "packets_total": [10, 10],
            "lost_packets": [0, 1],
            "latency_ms": [30.0, 31.0],
            "retries": [0, 0],
            "connection_resets": [0, 0],
            "target": [0, 0],
        }
    )

    prepared = build_sensor_metrics_frame(raw_frame)

    assert list(prepared.columns) == [
        "session_id",
        "timestamp",
        "packets_sent",
        "packets_lost",
        "rtt_ms",
        "retransmissions",
        "resets",
        "label",
    ]
    assert prepared["session_id"].tolist() == ["f1", "f1"]


def test_build_sensor_metrics_frame_rejects_missing_columns() -> None:
    raw_frame = pd.DataFrame({"session_id": ["s1"]})

    try:
        build_sensor_metrics_frame(raw_frame)
    except ValueError as exc:
        assert "Missing required source column" in str(exc)
    else:
        raise AssertionError("expected ValueError")
