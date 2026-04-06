from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np

from ml.training.morpher_model import MorpherConfig, build_checkpoint_payload, build_morpher_model
from ml.training.morpher_pipeline import build_iat_sequence_dataset, load_iat_table
from ml.training.sensor_metrics import compute_regression_metrics


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the ChameleonFlow traffic morpher.")
    parser.add_argument("--input", required=True, type=Path, help="IAT CSV/Parquet file")
    parser.add_argument("--output-model", required=True, type=Path, help="Torch checkpoint path")
    parser.add_argument(
        "--output-metadata",
        required=True,
        type=Path,
        help="Training metadata JSON output path",
    )
    parser.add_argument("--output-onnx", type=Path, default=None)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def _require_torch() -> tuple[Any, Any, Any, Any, Any]:
    import torch
    from torch import nn
    from torch.optim import Adam
    from torch.utils.data import DataLoader, TensorDataset

    return torch, nn, Adam, DataLoader, TensorDataset


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


def _set_random_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_dataset(
    dataset: Any,
    *,
    validation_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= validation_ratio < 1.0:
        msg = "validation_ratio must be in the range [0.0, 1.0)."
        raise ValueError(msg)

    indices = np.arange(dataset.inputs.shape[0])
    if len(indices) < 2 or validation_ratio == 0.0:
        return indices, np.empty(shape=(0,), dtype=np.int64)

    validation_size = max(1, int(math.floor(len(indices) * validation_ratio)))
    validation_indices = indices[-validation_size:]
    training_indices = indices[:-validation_size]
    if len(training_indices) == 0:
        msg = "validation_ratio leaves no rows for training."
        raise ValueError(msg)

    return training_indices, validation_indices


def _evaluate_model(
    torch: Any,
    model: Any,
    loss_fn: Any,
    loader: Any,
    device: Any,
    *,
    normalization_mean: float,
    normalization_std: float,
) -> dict[str, float] | None:
    if len(loader.dataset) == 0:
        return None

    total_loss = 0.0
    total_examples = 0
    predictions_batches: list[np.ndarray] = []
    targets_batches: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            predictions = model(batch_inputs)
            loss = loss_fn(predictions, batch_targets)
            batch_size = int(batch_inputs.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            predictions_batches.append(predictions.detach().cpu().numpy())
            targets_batches.append(batch_targets.detach().cpu().numpy())

    model.train()
    stacked_predictions = np.concatenate(predictions_batches, axis=0).reshape(-1)
    stacked_targets = np.concatenate(targets_batches, axis=0).reshape(-1)
    denormalized_predictions = (stacked_predictions * normalization_std) + normalization_mean
    denormalized_targets = (stacked_targets * normalization_std) + normalization_mean
    regression_metrics = compute_regression_metrics(
        targets=denormalized_targets,
        predictions=denormalized_predictions,
    )
    return {
        "huber_loss": total_loss / total_examples,
        "mae_ms": regression_metrics["mae"],
        "rmse_ms": regression_metrics["rmse"],
    }


def train_morpher(
    *,
    input_path: Path,
    output_model_path: Path,
    output_metadata_path: Path,
    output_onnx_path: Path | None,
    sequence_length: int,
    hidden_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    validation_ratio: float,
    seed: int,
    requested_device: str,
) -> dict[str, Any]:
    torch, nn, Adam, DataLoader, TensorDataset = _require_torch()
    _set_random_seed(torch, seed)
    device = _resolve_device(torch, requested_device)

    raw_frame = load_iat_table(input_path)
    dataset = build_iat_sequence_dataset(raw_frame, sequence_length=sequence_length)
    train_indices, validation_indices = _split_dataset(dataset, validation_ratio=validation_ratio)
    config = MorpherConfig(sequence_length=sequence_length, hidden_size=hidden_size)

    model = build_morpher_model(nn, config).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.HuberLoss()

    training_dataset = TensorDataset(
        torch.from_numpy(dataset.inputs[train_indices]),
        torch.from_numpy(dataset.targets[train_indices]),
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(dataset.inputs[validation_indices]),
        torch.from_numpy(dataset.targets[validation_indices]),
    )
    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    final_loss = 0.0
    best_validation_metrics: dict[str, float] | None = None
    model.train()
    for _ in range(epochs):
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = loss_fn(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())
        validation_metrics = _evaluate_model(
            torch,
            model,
            loss_fn,
            validation_loader,
            device,
            normalization_mean=dataset.mean,
            normalization_std=dataset.std,
        )
        if validation_metrics is not None:
            if (
                best_validation_metrics is None
                or validation_metrics["huber_loss"] < best_validation_metrics["huber_loss"]
            ):
                best_validation_metrics = validation_metrics

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = build_checkpoint_payload(
        config=config,
        state_dict=model.state_dict(),
        normalization_mean=dataset.mean,
        normalization_std=dataset.std,
    )
    torch.save(checkpoint, output_model_path)

    metadata = {
        "model_type": "lstm",
        "sequence_length": sequence_length,
        "hidden_size": hidden_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": str(device),
        "seed": seed,
        "training_examples": int(len(train_indices)),
        "validation_examples": int(len(validation_indices)),
        "normalization_mean": dataset.mean,
        "normalization_std": dataset.std,
        "final_train_loss": final_loss,
        "validation_metrics": best_validation_metrics,
        "onnx_exported": False,
    }

    if output_onnx_path is not None:
        output_onnx_path.parent.mkdir(parents=True, exist_ok=True)
        example_input = torch.from_numpy(dataset.inputs[:1]).to(device)
        torch.onnx.export(
            model,
            example_input,
            output_onnx_path,
            input_names=["iat_sequence"],
            output_names=["next_iat"],
            opset_version=17,
        )
        metadata["onnx_exported"] = True

    output_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    output_metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()
    metadata = train_morpher(
        input_path=args.input,
        output_model_path=args.output_model,
        output_metadata_path=args.output_metadata,
        output_onnx_path=args.output_onnx,
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        requested_device=args.device,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
