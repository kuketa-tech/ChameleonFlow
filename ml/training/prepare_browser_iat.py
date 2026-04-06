from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare browser IAT rows from packet timestamps.")
    parser.add_argument("--input", required=True, type=Path, help="Packet table CSV/Parquet")
    parser.add_argument("--output", required=True, type=Path, help="Prepared IAT CSV/Parquet")
    parser.add_argument("--trace-column", default="trace_id")
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--packet-index-column", default="packet_index")
    parser.add_argument("--max-iat-ms", type=float, default=5_000.0)
    return parser


def load_packet_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)

    if path.suffix == ".parquet":
        return pd.read_parquet(path)

    msg = f"Unsupported file format: {path.suffix}"
    raise ValueError(msg)


def _parse_timestamp_column(column: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(column):
        return pd.to_datetime(column.astype(float), unit="s", utc=True)

    return pd.to_datetime(column, utc=True, format="mixed")


def build_browser_iat_frame(
    raw_frame: pd.DataFrame,
    *,
    trace_column: str = "trace_id",
    timestamp_column: str = "timestamp",
    packet_index_column: str = "packet_index",
    max_iat_ms: float = 5_000.0,
) -> pd.DataFrame:
    required_columns = {trace_column, timestamp_column}
    missing = required_columns - set(raw_frame.columns)
    if missing:
        msg = f"Missing required packet columns: {sorted(missing)}"
        raise ValueError(msg)

    frame = raw_frame.copy()
    frame[timestamp_column] = _parse_timestamp_column(frame[timestamp_column])

    sort_columns = [trace_column]
    if packet_index_column in frame.columns:
        sort_columns.append(packet_index_column)
    else:
        sort_columns.append(timestamp_column)

    frame = frame.sort_values(sort_columns).reset_index(drop=True)
    frame["iat_ms"] = (
        frame.groupby(trace_column)[timestamp_column]
        .diff()
        .dt.total_seconds()
        .mul(1000.0)
    )

    prepared = frame.loc[
        frame["iat_ms"].notna()
        & (frame["iat_ms"] > 0.0)
        & (frame["iat_ms"] <= max_iat_ms),
        [trace_column, "iat_ms"],
    ].copy()
    prepared = prepared.rename(columns={trace_column: "trace_id"})

    if packet_index_column in frame.columns:
        prepared["packet_index"] = frame.loc[prepared.index, packet_index_column].to_numpy()
        prepared = prepared[["trace_id", "packet_index", "iat_ms"]]

    return prepared.reset_index(drop=True)


def save_iat_frame(frame: pd.DataFrame, output_path: Path) -> None:
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
    packet_table = load_packet_table(args.input)
    iat_frame = build_browser_iat_frame(
        packet_table,
        trace_column=args.trace_column,
        timestamp_column=args.timestamp_column,
        packet_index_column=args.packet_index_column,
        max_iat_ms=args.max_iat_ms,
    )
    save_iat_frame(iat_frame, args.output)


if __name__ == "__main__":
    main()
