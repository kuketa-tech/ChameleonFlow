from __future__ import annotations

import pandas as pd

from ml.training.prepare_browser_iat import build_browser_iat_frame


def test_build_browser_iat_frame_calculates_intervals_per_trace() -> None:
    raw_frame = pd.DataFrame(
        {
            "trace_id": ["a", "a", "a", "b", "b"],
            "packet_index": [0, 1, 2, 0, 1],
            "timestamp": [
                "2026-04-06T10:00:00Z",
                "2026-04-06T10:00:00.050Z",
                "2026-04-06T10:00:00.150Z",
                "2026-04-06T10:01:00Z",
                "2026-04-06T10:01:00.030Z",
            ],
        }
    )

    iat_frame = build_browser_iat_frame(raw_frame)

    assert iat_frame.to_dict(orient="records") == [
        {"trace_id": "a", "packet_index": 1, "iat_ms": 50.0},
        {"trace_id": "a", "packet_index": 2, "iat_ms": 100.0},
        {"trace_id": "b", "packet_index": 1, "iat_ms": 30.0},
    ]


def test_build_browser_iat_frame_filters_non_positive_and_large_intervals() -> None:
    raw_frame = pd.DataFrame(
        {
            "trace_id": ["a", "a", "a"],
            "packet_index": [0, 1, 2],
            "timestamp": [0.0, 0.0, 10.0],
        }
    )

    iat_frame = build_browser_iat_frame(raw_frame, max_iat_ms=1_000.0)

    assert iat_frame.empty
