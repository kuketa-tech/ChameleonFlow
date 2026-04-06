from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "packet_loss_ratio",
    "packets_per_second",
    "packets_total_log1p",
    "packets_lost_log1p",
    "rtt_mean_ms",
    "rtt_cv",
    "retransmission_ratio",
    "retransmissions_log1p",
    "reset_ratio",
    "sample_count",
    "flow_duration_ms",
    "flow_packets_per_second",
    "flow_bytes_per_second",
    "average_packet_size",
    "packet_length_cv",
    "fwd_backward_packet_ratio",
    "fwd_backward_byte_ratio",
    "down_up_ratio",
    "flow_iat_cv",
    "active_idle_ratio",
    "syn_flag_ratio",
    "ack_flag_ratio",
    "psh_flag_ratio",
]

RAW_SENSOR_COLUMN_LIST = [
    "session_id",
    "timestamp",
    "packets_sent",
    "packets_lost",
    "rtt_ms",
    "retransmissions",
    "resets",
    "label",
]
RAW_SENSOR_COLUMNS = set(RAW_SENSOR_COLUMN_LIST)

OPTIONAL_SENSOR_COLUMN_AGGREGATIONS = {
    "flow_duration_ms": "mean",
    "flow_packets_per_second": "mean",
    "flow_bytes_per_second": "mean",
    "average_packet_size": "mean",
    "packet_length_mean": "mean",
    "packet_length_std": "mean",
    "fwd_packet_count": "sum",
    "bwd_packet_count": "sum",
    "fwd_byte_count": "sum",
    "bwd_byte_count": "sum",
    "down_up_ratio": "mean",
    "flow_iat_mean_us": "mean",
    "flow_iat_std_us": "mean",
    "active_mean_us": "mean",
    "idle_mean_us": "mean",
    "syn_flag_count": "sum",
    "ack_flag_count": "sum",
    "psh_flag_count": "sum",
}


def _column_or_zeros(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    return pd.Series(0.0, index=frame.index, dtype="float64")


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)

    if path.suffix == ".parquet":
        return pd.read_parquet(path)

    msg = f"Unsupported file format: {path.suffix}"
    raise ValueError(msg)


def build_sensor_feature_frame(
    raw_frame: pd.DataFrame,
    *,
    window_seconds: int = 5,
) -> pd.DataFrame:
    missing = RAW_SENSOR_COLUMNS - set(raw_frame.columns)
    if missing:
        msg = f"Missing required sensor columns: {sorted(missing)}"
        raise ValueError(msg)

    frame = raw_frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["window_start"] = frame["timestamp"].dt.floor(f"{window_seconds}s")

    aggregations: dict[str, tuple[str, str]] = {
        "packets_sent": ("packets_sent", "sum"),
        "packets_lost": ("packets_lost", "sum"),
        "rtt_mean": ("rtt_ms", "mean"),
        "rtt_std": ("rtt_ms", "std"),
        "retransmissions": ("retransmissions", "sum"),
        "resets": ("resets", "sum"),
        "sample_count": ("timestamp", "count"),
        "label": ("label", "max"),
    }
    for column, aggregation in OPTIONAL_SENSOR_COLUMN_AGGREGATIONS.items():
        if column in frame.columns:
            aggregations[column] = (column, aggregation)

    grouped = frame.groupby(["session_id", "window_start"], as_index=False).agg(**aggregations)
    grouped = grouped.fillna({"rtt_std": 0.0})

    total_packets = (grouped["packets_sent"] + grouped["packets_lost"]).clip(lower=1)
    rtt_mean = grouped["rtt_mean"].clip(lower=1e-6)
    sample_count = grouped["sample_count"].clip(lower=1)
    window_seconds_float = float(max(window_seconds, 1))

    packet_length_mean = _column_or_zeros(grouped, "packet_length_mean").clip(lower=1e-6)
    packet_length_std = _column_or_zeros(grouped, "packet_length_std")
    bwd_packet_count = _column_or_zeros(grouped, "bwd_packet_count").clip(lower=1.0)
    bwd_byte_count = _column_or_zeros(grouped, "bwd_byte_count").clip(lower=1.0)
    flow_iat_mean_us = _column_or_zeros(grouped, "flow_iat_mean_us").clip(lower=1.0)
    idle_mean_us = _column_or_zeros(grouped, "idle_mean_us").clip(lower=1.0)

    feature_frame = grouped.assign(
        packet_loss_ratio=grouped["packets_lost"] / total_packets,
        packets_per_second=total_packets / window_seconds_float,
        packets_total_log1p=np.log1p(total_packets),
        packets_lost_log1p=np.log1p(grouped["packets_lost"]),
        rtt_mean_ms=grouped["rtt_mean"],
        rtt_cv=grouped["rtt_std"] / rtt_mean,
        retransmission_ratio=grouped["retransmissions"] / total_packets,
        retransmissions_log1p=np.log1p(grouped["retransmissions"]),
        reset_ratio=grouped["resets"] / sample_count,
        flow_duration_ms=_column_or_zeros(grouped, "flow_duration_ms"),
        flow_packets_per_second=_column_or_zeros(grouped, "flow_packets_per_second"),
        flow_bytes_per_second=_column_or_zeros(grouped, "flow_bytes_per_second"),
        average_packet_size=_column_or_zeros(grouped, "average_packet_size"),
        packet_length_cv=packet_length_std / packet_length_mean,
        fwd_backward_packet_ratio=_column_or_zeros(grouped, "fwd_packet_count") / bwd_packet_count,
        fwd_backward_byte_ratio=_column_or_zeros(grouped, "fwd_byte_count") / bwd_byte_count,
        down_up_ratio=_column_or_zeros(grouped, "down_up_ratio"),
        flow_iat_cv=_column_or_zeros(grouped, "flow_iat_std_us") / flow_iat_mean_us,
        active_idle_ratio=_column_or_zeros(grouped, "active_mean_us") / idle_mean_us,
        syn_flag_ratio=_column_or_zeros(grouped, "syn_flag_count") / total_packets,
        ack_flag_ratio=_column_or_zeros(grouped, "ack_flag_count") / total_packets,
        psh_flag_ratio=_column_or_zeros(grouped, "psh_flag_count") / total_packets,
    )[
        [
            "session_id",
            "window_start",
            *FEATURE_COLUMNS,
            "label",
        ]
    ]

    return feature_frame
