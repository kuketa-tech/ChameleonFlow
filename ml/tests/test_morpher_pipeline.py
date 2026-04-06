from __future__ import annotations

import pandas as pd

from ml.training.morpher_pipeline import build_iat_sequence_dataset


def test_build_iat_sequence_dataset_creates_shifted_examples() -> None:
    raw_frame = pd.DataFrame(
        {
            "trace_id": ["a"] * 5 + ["b"] * 5,
            "packet_index": [0, 1, 2, 3, 4] * 2,
            "iat_ms": [10, 20, 30, 40, 50, 15, 25, 35, 45, 55],
        }
    )

    dataset = build_iat_sequence_dataset(raw_frame, sequence_length=3)

    assert dataset.inputs.shape == (4, 3, 1)
    assert dataset.targets.shape == (4, 1)
    assert dataset.std > 0


def test_build_iat_sequence_dataset_requires_long_enough_trace() -> None:
    raw_frame = pd.DataFrame(
        {
            "trace_id": ["a", "a", "a"],
            "iat_ms": [10, 20, 30],
        }
    )

    try:
        build_iat_sequence_dataset(raw_frame, sequence_length=3)
    except ValueError as exc:
        assert "sequence_length" in str(exc)
    else:
        raise AssertionError("expected ValueError")
