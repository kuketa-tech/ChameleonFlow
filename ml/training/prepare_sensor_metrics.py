from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from ml.training.sensor_pipeline import (
    OPTIONAL_SENSOR_COLUMN_AGGREGATIONS,
    RAW_SENSOR_COLUMN_LIST,
    RAW_SENSOR_COLUMNS,
    load_table,
)

ALIASES = {
    "session_id": ["session_id", "flow_id", "trace_id", "connection_id"],
    "timestamp": ["timestamp", "time", "ts"],
    "packets_sent": ["packets_sent", "packets_total", "packets", "tx_packets"],
    "packets_lost": ["packets_lost", "lost_packets", "packet_loss_count"],
    "rtt_ms": ["rtt_ms", "rtt", "latency_ms"],
    "retransmissions": ["retransmissions", "retransmits", "retries"],
    "resets": ["resets", "connection_resets", "rst_count"],
    "label": ["label", "degraded", "target"],
}


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize sensor metric tables to training format.")
    parser.add_argument("--input", required=True, type=Path, help="Input CSV/Parquet")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV/Parquet")
    return parser


def _resolve_source_column(frame: pd.DataFrame, canonical_name: str) -> str:
    for candidate in ALIASES[canonical_name]:
        if candidate in frame.columns:
            return candidate

    msg = f"Missing required source column for {canonical_name}: {ALIASES[canonical_name]}"
    raise ValueError(msg)


def build_sensor_metrics_frame(
    raw_frame: pd.DataFrame,
    *,
    require_label: bool = True,
    passthrough_columns: Sequence[str] = (),
) -> pd.DataFrame:
    renamed_columns: dict[str, str] = {}
    required_columns = sorted(RAW_SENSOR_COLUMNS - ({ "label"} if not require_label else set()))
    for canonical_name in required_columns:
        source_column = _resolve_source_column(raw_frame, canonical_name)
        renamed_columns[source_column] = canonical_name

    frame = raw_frame.rename(columns=renamed_columns).copy()
    if require_label:
        source_label_column = _resolve_source_column(raw_frame, "label")
        frame = frame.rename(columns={source_label_column: "label"})
    else:
        frame["label"] = 0

    optional_columns = [
        column for column in OPTIONAL_SENSOR_COLUMN_AGGREGATIONS if column in frame.columns
    ]
    passthrough_list = [
        column
        for column in passthrough_columns
        if column in frame.columns and column not in { *RAW_SENSOR_COLUMN_LIST, *optional_columns}
    ]
    prepared = frame[[*RAW_SENSOR_COLUMN_LIST, *optional_columns, *passthrough_list]].copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], utc=True, format="mixed")
    prepared["label"] = prepared["label"].astype(int)

    numeric_columns = [
        "packets_sent",
        "packets_lost",
        "rtt_ms",
        "retransmissions",
        "resets",
        *optional_columns,
    ]
    for column in numeric_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="raise")

    return prepared.sort_values(["session_id", "timestamp"]).reset_index(drop=True)


def save_sensor_metrics_frame(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".csv":
        frame.to_csv(output_path, index=False)
        return

    if output_path.suffix == ".parquet":
        frame.to_parquet(output_path, index=False)
        return

    msg = f"Unsupported file format: {output_path.suffix}"
    raise ValueError(msg)


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()
    raw_frame = load_table(args.input)
    prepared = build_sensor_metrics_frame(raw_frame)
    save_sensor_metrics_frame(prepared, args.output)


if __name__ == "__main__":
    main()
