from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.training.morpher_model import MorpherConfig, build_morpher_model
from ml.training.morpher_pipeline import RAW_MORPHER_COLUMNS, load_iat_table


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ChameleonFlow morpher inference on IAT sequences.")
    parser.add_argument("--input", required=True, type=Path, help="IAT CSV/Parquet")
    parser.add_argument("--model", required=True, type=Path, help="Torch checkpoint path")
    parser.add_argument("--output", required=True, type=Path, help="Predictions CSV/Parquet")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def _require_torch() -> tuple[Any, Any]:
    import torch
    from torch import nn

    return torch, nn


def _resolve_device(torch: Any, requested_device: str) -> Any:
    if requested_device == "cpu":
        return torch.device("cpu")

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            msg = "CUDA was requested explicitly, but torch.cuda.is_available() is False."
            raise RuntimeError(msg)
        return torch.device("cuda")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def _build_inference_inputs(
    raw_frame: pd.DataFrame,
    *,
    sequence_length: int,
    mean: float,
    std: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    missing = RAW_MORPHER_COLUMNS - set(raw_frame.columns)
    if missing:
        msg = f"Missing required morpher columns: {sorted(missing)}"
        raise ValueError(msg)

    sort_columns = ["trace_id"]
    if "packet_index" in raw_frame.columns:
        sort_columns.append("packet_index")

    frame = raw_frame.sort_values(sort_columns).reset_index(drop=True)
    rows_meta: list[dict[str, Any]] = []
    inputs: list[np.ndarray] = []

    for trace_id, trace_frame in frame.groupby("trace_id", sort=False):
        trace_iat = trace_frame["iat_ms"].astype(float).to_numpy()
        packet_indexes = (
            trace_frame["packet_index"].to_numpy()
            if "packet_index" in trace_frame.columns
            else np.arange(len(trace_frame))
        )
        if len(trace_iat) <= sequence_length:
            continue

        normalized = (trace_iat - mean) / std
        for index in range(sequence_length, len(normalized)):
            inputs.append(normalized[index - sequence_length : index])
            rows_meta.append(
                {
                    "trace_id": trace_id,
                    "target_packet_index": int(packet_indexes[index]),
                    "actual_iat_ms": float(trace_iat[index]),
                }
            )

    if not inputs:
        msg = "Morpher inference requires at least one trace longer than sequence_length."
        raise ValueError(msg)

    return np.asarray(inputs, dtype=np.float32)[:, :, None], pd.DataFrame(rows_meta)


def save_prediction_frame(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".csv":
        frame.to_csv(output_path, index=False)
        return

    if output_path.suffix == ".parquet":
        frame.to_parquet(output_path, index=False)
        return

    msg = f"Unsupported output format: {output_path.suffix}"
    raise ValueError(msg)


def infer_morpher(
    *,
    input_path: Path,
    model_path: Path,
    output_path: Path,
    requested_device: str,
) -> dict[str, Any]:
    torch, nn = _require_torch()
    device = _resolve_device(torch, requested_device)
    checkpoint = torch.load(model_path, map_location=device)

    config = MorpherConfig(**checkpoint["config"])
    mean = float(checkpoint["normalization"]["mean"])
    std = float(checkpoint["normalization"]["std"]) or 1.0

    model = build_morpher_model(nn, config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    raw_frame = load_iat_table(input_path)
    inputs, prediction_meta = _build_inference_inputs(
        raw_frame,
        sequence_length=config.sequence_length,
        mean=mean,
        std=std,
    )

    with torch.no_grad():
        tensor_inputs = torch.from_numpy(inputs).to(device)
        predictions = model(tensor_inputs).detach().cpu().numpy().reshape(-1)

    denormalized_predictions = (predictions * std) + mean
    prediction_meta["predicted_iat_ms"] = denormalized_predictions
    prediction_meta["abs_error_ms"] = (
        prediction_meta["predicted_iat_ms"] - prediction_meta["actual_iat_ms"]
    ).abs()
    save_prediction_frame(prediction_meta, output_path)

    return {
        "rows": int(len(prediction_meta)),
        "device": str(device),
        "sequence_length": config.sequence_length,
        "hidden_size": config.hidden_size,
    }


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()
    summary = infer_morpher(
        input_path=args.input,
        model_path=args.model,
        output_path=args.output,
        requested_device=args.device,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
