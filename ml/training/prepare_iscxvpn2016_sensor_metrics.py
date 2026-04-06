from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff


INPUT_ALIASES = {
    "session_id": ["session_id", "flow_id", "flowid", "id", "fid"],
    "label": ["label", "class", "category", "traffic_type", "service", "app"],
    "total_packets": ["total_packets", "total_packets_count", "packets", "total_pkt"],
    "total_fwd_packets": [
        "total_fwd_packets",
        "total_forward_packets",
        "total_fpackets",
        "fwd_packet_count",
        "subflow_fwd_packets",
    ],
    "total_bwd_packets": [
        "total_backward_packets",
        "total_bwd_packets",
        "total_bpackets",
        "bwd_packet_count",
        "subflow_bwd_packets",
    ],
    "total_fwd_bytes": [
        "total_length_of_fwd_packets",
        "total_fwd_bytes",
        "total_fvolume",
        "subflow_fwd_bytes",
    ],
    "total_bwd_bytes": [
        "total_length_of_bwd_packets",
        "total_bwd_bytes",
        "total_bvolume",
        "subflow_bwd_bytes",
    ],
    "rst_flag_count": ["rst_flag_count", "rst_cnt", "reset_count"],
    "syn_flag_count": ["syn_flag_count", "syn_cnt"],
    "ack_flag_count": ["ack_flag_count", "ack_cnt"],
    "psh_flag_count": ["psh_flag_count", "psh_cnt"],
    "flow_duration_us": ["flow_duration", "duration", "dur"],
    "flow_duration_ms": ["flow_duration_ms", "duration_ms"],
    "flow_packets_per_second": ["flow_packets_s", "flow_packets_per_second", "flow_pkts_s"],
    "flow_bytes_per_second": ["flow_bytes_s", "flow_bytes_per_second"],
    "flow_iat_mean_us": ["flow_iat_mean", "iat_mean", "flowiatmean"],
    "flow_iat_std_us": ["flow_iat_std", "iat_std", "flowiatstd"],
    "flow_iat_mean_ms": ["flow_iat_mean_ms", "iat_mean_ms"],
    "flow_iat_std_ms": ["flow_iat_std_ms", "iat_std_ms"],
    "average_packet_size": ["average_packet_size", "pkt_size_avg", "packet_size_mean"],
    "packet_length_mean": ["packet_length_mean", "pkt_len_mean", "packet_size_mean"],
    "packet_length_std": ["packet_length_std", "pkt_len_std", "packet_size_std"],
    "down_up_ratio": ["down_up_ratio", "down_up_ratio_count"],
    "active_mean_us": ["active_mean", "active_mean_us"],
    "idle_mean_us": ["idle_mean", "idle_mean_us"],
    "act_data_pkt_fwd": ["act_data_pkt_fwd", "actual_data_pkt_fwd"],
    "retransmissions": ["retransmissions", "retransmission_count", "retries"],
}

INPUT_ALIASES["flow_packets_per_second"].extend(["flowpktspersecond"])
INPUT_ALIASES["flow_bytes_per_second"].extend(["flowbytespersecond"])
INPUT_ALIASES["flow_iat_mean_us"].extend(["mean_flowiat"])
INPUT_ALIASES["flow_iat_std_us"].extend(["std_flowiat"])
INPUT_ALIASES["active_mean_us"].extend(["mean_active"])
INPUT_ALIASES["idle_mean_us"].extend(["mean_idle"])


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert ISCXVPN2016 flow tables to sensor_metrics.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        type=Path,
        help="One or more ISCXVPN2016 CSV/ARFF files",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output CSV/Parquet")
    parser.add_argument(
        "--positive-pattern",
        default="vpn",
        help="Substring pattern that marks a row as degraded/positive when labels are textual",
    )
    return parser


def _canonicalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed_columns: dict[str, str] = {}
    seen: set[str] = set()
    for original_name in frame.columns:
        base_name = _canonicalize_name(str(original_name))
        candidate = base_name or "column"
        suffix = 2
        while candidate in seen:
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        seen.add(candidate)
        renamed_columns[str(original_name)] = candidate

    normalized = frame.rename(columns=renamed_columns).copy()
    for column in normalized.columns:
        if normalized[column].dtype == object:
            normalized[column] = normalized[column].map(
                lambda value: value.decode("utf-8", errors="replace")
                if isinstance(value, bytes)
                else value
            )
    return normalized


def load_iscxvpn2016_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".arff":
        try:
            data, _ = arff.loadarff(path)
            return pd.DataFrame(data)
        except NotImplementedError:
            return _load_arff_fallback(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    msg = f"Unsupported ISCXVPN2016 file format: {path.suffix}"
    raise ValueError(msg)


def _load_arff_fallback(path: Path) -> pd.DataFrame:
    attribute_names: list[str] = []
    data_lines: list[str] = []
    in_data_section = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue

        lower = line.lower()
        if lower.startswith("@attribute"):
            match = re.match(r"@attribute\s+('.*?'|\".*?\"|[^\s]+)\s+.+", line, flags=re.IGNORECASE)
            if match is None:
                msg = f"Invalid ARFF attribute declaration: {line}"
                raise ValueError(msg)
            attribute_name = match.group(1).strip("\"'")
            attribute_names.append(attribute_name)
            continue

        if lower.startswith("@data"):
            in_data_section = True
            continue

        if in_data_section:
            data_lines.append(line)

    reader = csv.reader(data_lines, delimiter=",", quotechar="'", skipinitialspace=True)
    rows = list(reader)
    return pd.DataFrame(rows, columns=attribute_names)


def _resolve_column(frame: pd.DataFrame, key: str) -> str | None:
    for candidate in INPUT_ALIASES[key]:
        if candidate in frame.columns:
            return candidate

    return None


def _series_or_default(frame: pd.DataFrame, key: str, *, default: float = 0.0) -> pd.Series:
    column = _resolve_column(frame, key)
    if column is None:
        return pd.Series(default, index=frame.index, dtype="float64")

    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _build_labels(frame: pd.DataFrame, *, source_name: str, positive_pattern: str) -> pd.Series:
    label_column = _resolve_column(frame, "label")
    if label_column is None:
        label_values = pd.Series(source_name, index=frame.index, dtype="object")
    else:
        label_values = frame[label_column].astype(str)

    numeric_labels = pd.to_numeric(label_values, errors="coerce")
    if numeric_labels.notna().all():
        unique_numeric = set(numeric_labels.astype(int).unique().tolist())
        if unique_numeric <= {0, 1}:
            return numeric_labels.astype(int)

    pattern = positive_pattern.strip().lower()
    if not pattern:
        return pd.Series(0, index=frame.index, dtype="int64")

    normalized = label_values.str.lower()
    positives = normalized.str.contains(pattern, regex=False)
    negatives = normalized.str.contains(f"non{pattern}", regex=False)
    return (positives & ~negatives).astype(int)


def _build_single_frame(
    raw_frame: pd.DataFrame,
    *,
    source_name: str,
    positive_pattern: str,
) -> pd.DataFrame:
    frame = _normalize_columns(raw_frame)

    total_packets = _series_or_default(frame, "total_packets")
    total_fwd_packets = _series_or_default(frame, "total_fwd_packets")
    total_bwd_packets = _series_or_default(frame, "total_bwd_packets")
    if (total_fwd_packets == 0).all() and (total_bwd_packets == 0).all():
        total_fwd_packets = total_packets
    total_packets = (total_fwd_packets + total_bwd_packets).clip(lower=1)

    total_fwd_bytes = _series_or_default(frame, "total_fwd_bytes")
    total_bwd_bytes = _series_or_default(frame, "total_bwd_bytes")

    rst_flag_count = _series_or_default(frame, "rst_flag_count")
    syn_flag_count = _series_or_default(frame, "syn_flag_count")
    ack_flag_count = _series_or_default(frame, "ack_flag_count")
    psh_flag_count = _series_or_default(frame, "psh_flag_count")

    flow_duration_ms = _series_or_default(frame, "flow_duration_ms")
    if (flow_duration_ms == 0).all():
        flow_duration_ms = _series_or_default(frame, "flow_duration_us") / 1000.0

    flow_iat_mean_ms = _series_or_default(frame, "flow_iat_mean_ms")
    if (flow_iat_mean_ms == 0).all():
        flow_iat_mean_ms = _series_or_default(frame, "flow_iat_mean_us") / 1000.0

    flow_iat_std_us = _series_or_default(frame, "flow_iat_std_us")
    if (flow_iat_std_us == 0).all():
        flow_iat_std_us = _series_or_default(frame, "flow_iat_std_ms") * 1000.0

    retransmission_count = _series_or_default(frame, "retransmissions")
    if (retransmission_count == 0).all():
        act_data_pkt_fwd = _series_or_default(frame, "act_data_pkt_fwd")
        retransmission_count = (total_fwd_packets - act_data_pkt_fwd).clip(lower=0)

    average_packet_size = _series_or_default(frame, "average_packet_size")
    packet_length_mean = _series_or_default(frame, "packet_length_mean")
    if (packet_length_mean == 0).all():
        packet_length_mean = average_packet_size
    packet_length_std = _series_or_default(frame, "packet_length_std")
    flow_packets_per_second = _series_or_default(frame, "flow_packets_per_second")
    flow_bytes_per_second = _series_or_default(frame, "flow_bytes_per_second")
    down_up_ratio = _series_or_default(frame, "down_up_ratio")
    active_mean_us = _series_or_default(frame, "active_mean_us")
    idle_mean_us = _series_or_default(frame, "idle_mean_us")
    labels = _build_labels(frame, source_name=source_name, positive_pattern=positive_pattern)

    session_column = _resolve_column(frame, "session_id")
    if session_column is None:
        session_ids = [f"{source_name}-{index:09d}" for index in range(len(frame))]
    else:
        session_ids = frame[session_column].astype(str).tolist()

    base_timestamp = pd.Timestamp("2026-04-06T00:00:00Z")
    timestamps = [base_timestamp + pd.Timedelta(seconds=index) for index in range(len(frame))]

    prepared = pd.DataFrame(
        {
            "session_id": session_ids,
            "timestamp": timestamps,
            "packets_sent": total_packets.astype(int),
            "packets_lost": rst_flag_count.clip(lower=0, upper=total_packets).astype(int),
            "rtt_ms": flow_iat_mean_ms.clip(lower=1.0).astype(float),
            "retransmissions": retransmission_count.clip(lower=0).astype(int),
            "resets": rst_flag_count.clip(lower=0).astype(int),
            "flow_duration_ms": flow_duration_ms.clip(lower=0).astype(float),
            "flow_packets_per_second": flow_packets_per_second.clip(lower=0).astype(float),
            "flow_bytes_per_second": flow_bytes_per_second.clip(lower=0).astype(float),
            "average_packet_size": average_packet_size.clip(lower=0).astype(float),
            "packet_length_mean": packet_length_mean.clip(lower=0).astype(float),
            "packet_length_std": packet_length_std.clip(lower=0).astype(float),
            "fwd_packet_count": total_fwd_packets.clip(lower=0).astype(float),
            "bwd_packet_count": total_bwd_packets.clip(lower=0).astype(float),
            "fwd_byte_count": total_fwd_bytes.clip(lower=0).astype(float),
            "bwd_byte_count": total_bwd_bytes.clip(lower=0).astype(float),
            "down_up_ratio": down_up_ratio.clip(lower=0).astype(float),
            "flow_iat_mean_us": (flow_iat_mean_ms * 1000.0).clip(lower=0).astype(float),
            "flow_iat_std_us": flow_iat_std_us.clip(lower=0).astype(float),
            "active_mean_us": active_mean_us.clip(lower=0).astype(float),
            "idle_mean_us": idle_mean_us.clip(lower=0).astype(float),
            "syn_flag_count": syn_flag_count.clip(lower=0).astype(float),
            "ack_flag_count": ack_flag_count.clip(lower=0).astype(float),
            "psh_flag_count": psh_flag_count.clip(lower=0).astype(float),
            "label": labels.astype(int),
        }
    )
    return prepared.sort_values(["session_id", "timestamp"]).reset_index(drop=True)


def build_iscxvpn2016_sensor_metrics(
    input_paths: list[Path],
    *,
    positive_pattern: str = "vpn",
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for input_path in input_paths:
        raw_frame = load_iscxvpn2016_table(input_path)
        frames.append(
            _build_single_frame(
                raw_frame,
                source_name=input_path.stem,
                positive_pattern=positive_pattern,
            )
        )

    if not frames:
        msg = "At least one input file is required."
        raise ValueError(msg)

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    from ml.training.prepare_sensor_metrics import save_sensor_metrics_frame

    parser = _build_argument_parser()
    args = parser.parse_args()
    prepared = build_iscxvpn2016_sensor_metrics(
        args.inputs,
        positive_pattern=args.positive_pattern,
    )
    save_sensor_metrics_frame(prepared, args.output)


if __name__ == "__main__":
    main()
