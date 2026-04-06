from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml.training.prepare_sensor_metrics import save_sensor_metrics_frame
from ml.training.sensor_pipeline import load_table


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert CIC-IDS2017 flow CSV files to sensor_metrics.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        type=Path,
        help="One or more CIC-IDS2017 CSV files",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output CSV/Parquet")
    return parser


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized_columns = {column: column.strip() for column in frame.columns}
    return frame.rename(columns=normalized_columns)


def _build_single_frame(raw_frame: pd.DataFrame, *, source_name: str) -> pd.DataFrame:
    frame = _normalize_columns(raw_frame)

    total_fwd_packets = pd.to_numeric(frame["Total Fwd Packets"], errors="coerce").fillna(0).clip(lower=0)
    total_bwd_packets = (
        pd.to_numeric(frame["Total Backward Packets"], errors="coerce").fillna(0).clip(lower=0)
    )
    total_packets = (total_fwd_packets + total_bwd_packets).clip(lower=1)
    rst_flags = pd.to_numeric(frame["RST Flag Count"], errors="coerce").fillna(0).clip(lower=0)
    flow_iat_mean_us = pd.to_numeric(frame["Flow IAT Mean"], errors="coerce").fillna(0).clip(lower=0)
    flow_iat_std_us = pd.to_numeric(frame["Flow IAT Std"], errors="coerce").fillna(0).clip(lower=0)
    act_data_pkt_fwd = pd.to_numeric(frame["act_data_pkt_fwd"], errors="coerce").fillna(0).clip(lower=0)
    total_fwd_bytes = (
        pd.to_numeric(frame["Total Length of Fwd Packets"], errors="coerce").fillna(0).clip(lower=0)
    )
    total_bwd_bytes = (
        pd.to_numeric(frame["Total Length of Bwd Packets"], errors="coerce").fillna(0).clip(lower=0)
    )
    packet_length_mean = (
        pd.to_numeric(frame["Packet Length Mean"], errors="coerce").fillna(0).clip(lower=0)
    )
    packet_length_std = (
        pd.to_numeric(frame["Packet Length Std"], errors="coerce").fillna(0).clip(lower=0)
    )
    flow_duration_ms = (
        pd.to_numeric(frame["Flow Duration"], errors="coerce").fillna(0).clip(lower=0) / 1000.0
    )
    flow_packets_per_second = (
        pd.to_numeric(frame["Flow Packets/s"], errors="coerce").replace([np.inf, -np.inf], 0).fillna(0)
    ).clip(lower=0)
    flow_bytes_per_second = (
        pd.to_numeric(frame["Flow Bytes/s"], errors="coerce").replace([np.inf, -np.inf], 0).fillna(0)
    ).clip(lower=0)
    average_packet_size = (
        pd.to_numeric(frame["Average Packet Size"], errors="coerce").fillna(0).clip(lower=0)
    )
    down_up_ratio = (
        pd.to_numeric(frame["Down/Up Ratio"], errors="coerce").replace([np.inf, -np.inf], 0).fillna(0)
    ).clip(lower=0)
    active_mean_us = pd.to_numeric(frame["Active Mean"], errors="coerce").fillna(0).clip(lower=0)
    idle_mean_us = pd.to_numeric(frame["Idle Mean"], errors="coerce").fillna(0).clip(lower=0)
    syn_flag_count = pd.to_numeric(frame["SYN Flag Count"], errors="coerce").fillna(0).clip(lower=0)
    ack_flag_count = pd.to_numeric(frame["ACK Flag Count"], errors="coerce").fillna(0).clip(lower=0)
    psh_flag_count = pd.to_numeric(frame["PSH Flag Count"], errors="coerce").fillna(0).clip(lower=0)

    retransmissions = (total_fwd_packets - act_data_pkt_fwd).clip(lower=0)
    packets_lost = rst_flags.clip(upper=total_packets)
    rtt_ms = (flow_iat_mean_us / 1000.0).clip(lower=1.0)
    labels = frame["Label"].astype(str).str.strip().str.upper().ne("BENIGN").astype(int)

    base_timestamp = pd.Timestamp("2026-04-06T00:00:00Z")
    timestamps = [base_timestamp + pd.Timedelta(seconds=index) for index in range(len(frame))]

    prepared = pd.DataFrame(
        {
            "session_id": [f"{source_name}-{index:09d}" for index in range(len(frame))],
            "timestamp": timestamps,
            "packets_sent": total_packets.astype(int),
            "packets_lost": packets_lost.astype(int),
            "rtt_ms": rtt_ms.astype(float),
            "retransmissions": retransmissions.astype(int),
            "resets": rst_flags.astype(int),
            "flow_duration_ms": flow_duration_ms.astype(float),
            "flow_packets_per_second": flow_packets_per_second.astype(float),
            "flow_bytes_per_second": flow_bytes_per_second.astype(float),
            "average_packet_size": average_packet_size.astype(float),
            "packet_length_mean": packet_length_mean.astype(float),
            "packet_length_std": packet_length_std.astype(float),
            "fwd_packet_count": total_fwd_packets.astype(float),
            "bwd_packet_count": total_bwd_packets.astype(float),
            "fwd_byte_count": total_fwd_bytes.astype(float),
            "bwd_byte_count": total_bwd_bytes.astype(float),
            "down_up_ratio": down_up_ratio.astype(float),
            "flow_iat_mean_us": flow_iat_mean_us.astype(float),
            "flow_iat_std_us": flow_iat_std_us.astype(float),
            "active_mean_us": active_mean_us.astype(float),
            "idle_mean_us": idle_mean_us.astype(float),
            "syn_flag_count": syn_flag_count.astype(float),
            "ack_flag_count": ack_flag_count.astype(float),
            "psh_flag_count": psh_flag_count.astype(float),
            "label": labels.astype(int),
        }
    )
    return prepared


def build_cicids2017_sensor_metrics(input_paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for input_path in input_paths:
        raw_frame = load_table(input_path)
        frames.append(_build_single_frame(raw_frame, source_name=input_path.stem))

    if not frames:
        msg = "At least one input file is required."
        raise ValueError(msg)

    combined = pd.concat(frames, ignore_index=True)
    return combined


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()
    prepared = build_cicids2017_sensor_metrics(args.inputs)
    save_sensor_metrics_frame(prepared, args.output)


if __name__ == "__main__":
    main()
