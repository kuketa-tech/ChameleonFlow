from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_MORPHER_COLUMNS = {"trace_id", "iat_ms"}


@dataclass(slots=True)
class SequenceDataset:
    inputs: np.ndarray
    targets: np.ndarray
    mean: float
    std: float


def load_iat_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)

    if path.suffix == ".parquet":
        return pd.read_parquet(path)

    msg = f"Unsupported file format: {path.suffix}"
    raise ValueError(msg)


def build_iat_sequence_dataset(
    raw_frame: pd.DataFrame,
    *,
    sequence_length: int = 20,
) -> SequenceDataset:
    missing = RAW_MORPHER_COLUMNS - set(raw_frame.columns)
    if missing:
        msg = f"Missing required morpher columns: {sorted(missing)}"
        raise ValueError(msg)

    sort_columns = ["trace_id"]
    if "packet_index" in raw_frame.columns:
        sort_columns.append("packet_index")

    frame = raw_frame.sort_values(sort_columns).reset_index(drop=True)
    iat_values = frame["iat_ms"].astype(float).to_numpy()
    mean = float(iat_values.mean())
    std = float(iat_values.std()) or 1.0

    inputs: list[np.ndarray] = []
    targets: list[float] = []

    for _, trace_frame in frame.groupby("trace_id", sort=False):
        trace_iat = trace_frame["iat_ms"].astype(float).to_numpy()
        if len(trace_iat) <= sequence_length:
            continue

        normalized = (trace_iat - mean) / std
        for index in range(sequence_length, len(normalized)):
            inputs.append(normalized[index - sequence_length : index])
            targets.append(float(normalized[index]))

    if not inputs:
        msg = "Morpher training requires at least one trace longer than sequence_length."
        raise ValueError(msg)

    return SequenceDataset(
        inputs=np.asarray(inputs, dtype=np.float32)[:, :, None],
        targets=np.asarray(targets, dtype=np.float32)[:, None],
        mean=mean,
        std=std,
    )
