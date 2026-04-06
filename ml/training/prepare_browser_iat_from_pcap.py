from __future__ import annotations

import argparse
import subprocess
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

from ml.training.prepare_browser_iat import save_iat_frame


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build browser_iat.csv from pcap or zip archives.")
    parser.add_argument("--input", required=True, type=Path, help="Input .pcap, .pcapng, or .zip")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV/Parquet")
    parser.add_argument("--max-iat-ms", type=float, default=5_000.0)
    return parser


def _list_capture_files(input_path: Path) -> list[Path]:
    if input_path.suffix in {".pcap", ".pcapng"}:
        return [input_path]

    msg = f"Unsupported capture format: {input_path.suffix}"
    raise ValueError(msg)


def _parse_tcpdump_timestamps(capture_path: Path) -> list[float]:
    result = subprocess.run(
        ["tcpdump", "-tt", "-n", "-r", str(capture_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    timestamps: list[float] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        first_token = line.split(maxsplit=1)[0]
        try:
            timestamps.append(float(first_token))
        except ValueError:
            continue
    return timestamps


def build_iat_frame_from_captures(capture_paths: list[Path], *, max_iat_ms: float) -> pd.DataFrame:
    rows: list[dict[str, str | int | float]] = []
    for capture_path in capture_paths:
        timestamps = _parse_tcpdump_timestamps(capture_path)
        trace_id = capture_path.stem
        for packet_index in range(1, len(timestamps)):
            iat_ms = round((timestamps[packet_index] - timestamps[packet_index - 1]) * 1000.0, 6)
            if 0.0 < iat_ms <= max_iat_ms:
                rows.append(
                    {
                        "trace_id": trace_id,
                        "packet_index": packet_index,
                        "iat_ms": iat_ms,
                    }
                )

    return pd.DataFrame(rows, columns=["trace_id", "packet_index", "iat_ms"])


def build_iat_frame_from_archive(input_path: Path, *, max_iat_ms: float) -> pd.DataFrame:
    if input_path.suffix != ".zip":
        return build_iat_frame_from_captures(_list_capture_files(input_path), max_iat_ms=max_iat_ms)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        with zipfile.ZipFile(input_path) as archive:
            archive.extractall(tmp_path)
        capture_paths = sorted(
            [
                path
                for path in tmp_path.rglob("*")
                if path.suffix in {".pcap", ".pcapng"}
            ]
        )
        return build_iat_frame_from_captures(capture_paths, max_iat_ms=max_iat_ms)


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()
    iat_frame = build_iat_frame_from_archive(args.input, max_iat_ms=args.max_iat_ms)
    save_iat_frame(iat_frame, args.output)


if __name__ == "__main__":
    main()
