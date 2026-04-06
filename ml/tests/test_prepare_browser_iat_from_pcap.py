from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.training import prepare_browser_iat_from_pcap as module


def test_build_iat_frame_from_captures_uses_tcpdump_timestamps(monkeypatch) -> None:
    def fake_parse(_capture_path: Path) -> list[float]:
        return [1.0, 1.05, 1.20]

    monkeypatch.setattr(module, "_parse_tcpdump_timestamps", fake_parse)

    frame = module.build_iat_frame_from_captures([Path("trace-a.pcap")], max_iat_ms=500.0)

    assert frame.to_dict(orient="records") == [
        {"trace_id": "trace-a", "packet_index": 1, "iat_ms": 50.0},
        {"trace_id": "trace-a", "packet_index": 2, "iat_ms": 150.0},
    ]


def test_build_iat_frame_from_captures_filters_large_gaps(monkeypatch) -> None:
    def fake_parse(_capture_path: Path) -> list[float]:
        return [1.0, 10.0]

    monkeypatch.setattr(module, "_parse_tcpdump_timestamps", fake_parse)

    frame = module.build_iat_frame_from_captures([Path("trace-a.pcap")], max_iat_ms=500.0)

    assert isinstance(frame, pd.DataFrame)
    assert frame.empty
