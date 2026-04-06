from __future__ import annotations

import pandas as pd

from ml.training.sensor_pipeline import build_sensor_feature_frame


def test_build_sensor_feature_frame_aggregates_5_second_windows() -> None:
    raw_frame = pd.DataFrame(
        [
            {
                "session_id": "s1",
                "timestamp": "2026-04-06T10:00:00Z",
                "packets_sent": 10,
                "packets_lost": 2,
                "rtt_ms": 100.0,
                "retransmissions": 1,
                "resets": 0,
                "flow_duration_ms": 10.0,
                "flow_packets_per_second": 4.0,
                "flow_bytes_per_second": 512.0,
                "average_packet_size": 64.0,
                "packet_length_mean": 60.0,
                "packet_length_std": 12.0,
                "fwd_packet_count": 10.0,
                "bwd_packet_count": 2.0,
                "fwd_byte_count": 500.0,
                "bwd_byte_count": 100.0,
                "down_up_ratio": 0.2,
                "flow_iat_mean_us": 50_000.0,
                "flow_iat_std_us": 10_000.0,
                "active_mean_us": 4_000.0,
                "idle_mean_us": 2_000.0,
                "syn_flag_count": 1.0,
                "ack_flag_count": 10.0,
                "psh_flag_count": 0.0,
                "label": 0,
            },
            {
                "session_id": "s1",
                "timestamp": "2026-04-06T10:00:04Z",
                "packets_sent": 5,
                "packets_lost": 1,
                "rtt_ms": 200.0,
                "retransmissions": 1,
                "resets": 1,
                "flow_duration_ms": 20.0,
                "flow_packets_per_second": 6.0,
                "flow_bytes_per_second": 768.0,
                "average_packet_size": 96.0,
                "packet_length_mean": 90.0,
                "packet_length_std": 18.0,
                "fwd_packet_count": 5.0,
                "bwd_packet_count": 1.0,
                "fwd_byte_count": 400.0,
                "bwd_byte_count": 50.0,
                "down_up_ratio": 0.5,
                "flow_iat_mean_us": 100_000.0,
                "flow_iat_std_us": 30_000.0,
                "active_mean_us": 3_000.0,
                "idle_mean_us": 1_000.0,
                "syn_flag_count": 2.0,
                "ack_flag_count": 5.0,
                "psh_flag_count": 1.0,
                "label": 1,
            },
        ]
    )

    feature_frame = build_sensor_feature_frame(raw_frame, window_seconds=5)

    assert len(feature_frame) == 1
    row = feature_frame.iloc[0]
    assert row["packet_loss_ratio"] == 3 / 18
    assert row["packets_per_second"] == 18 / 5
    assert row["rtt_mean_ms"] == 150.0
    assert row["retransmission_ratio"] == 2 / 18
    assert row["reset_ratio"] == 1 / 2
    assert row["flow_duration_ms"] == 15.0
    assert row["flow_packets_per_second"] == 5.0
    assert row["packet_length_cv"] == 15.0 / 75.0
    assert row["fwd_backward_packet_ratio"] == 7.5 / 1.5
    assert row["flow_iat_cv"] == 20_000.0 / 75_000.0
    assert row["active_idle_ratio"] == 3_500.0 / 1_500.0
    assert row["syn_flag_ratio"] == 3.0 / 18.0
    assert row["label"] == 1


def test_build_sensor_feature_frame_rejects_missing_columns() -> None:
    raw_frame = pd.DataFrame([{"session_id": "s1"}])

    try:
        build_sensor_feature_frame(raw_frame)
    except ValueError as exc:
        assert "Missing required sensor columns" in str(exc)
    else:
        raise AssertionError("expected ValueError")
