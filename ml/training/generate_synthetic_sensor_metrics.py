from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a synthetic sensor training dataset.")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV/Parquet file")
    parser.add_argument("--sessions", type=int, default=200)
    parser.add_argument("--rows-per-session", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def build_synthetic_sensor_metrics_frame(
    *,
    sessions: int = 200,
    rows_per_session: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, int | float | str]] = []
    base_time = pd.Timestamp("2026-04-06T10:00:00Z")

    for session_index in range(sessions):
        label = int(session_index >= sessions // 2)
        session_id = f"session-{session_index:05d}"
        for row_index in range(rows_per_session):
            timestamp = base_time + pd.Timedelta(seconds=(session_index * rows_per_session) + row_index)
            if label == 0:
                packets_sent = int(rng.integers(8, 15))
                packets_lost = int(rng.integers(0, 2))
                rtt_ms = float(rng.normal(35.0, 5.0))
                retransmissions = int(rng.integers(0, 1))
                resets = int(rng.integers(0, 1))
            else:
                packets_sent = int(rng.integers(8, 15))
                packets_lost = int(rng.integers(3, 10))
                rtt_ms = float(rng.normal(180.0, 30.0))
                retransmissions = int(rng.integers(1, 5))
                resets = int(rng.integers(0, 2))

            rows.append(
                {
                    "session_id": session_id,
                    "timestamp": timestamp.isoformat(),
                    "packets_sent": packets_sent,
                    "packets_lost": packets_lost,
                    "rtt_ms": max(rtt_ms, 1.0),
                    "retransmissions": retransmissions,
                    "resets": resets,
                    "label": label,
                }
            )

    return pd.DataFrame(rows)


def save_synthetic_sensor_metrics(frame: pd.DataFrame, output_path: Path) -> None:
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
    frame = build_synthetic_sensor_metrics_frame(
        sessions=args.sessions,
        rows_per_session=args.rows_per_session,
        seed=args.seed,
    )
    save_synthetic_sensor_metrics(frame, args.output)


if __name__ == "__main__":
    main()
