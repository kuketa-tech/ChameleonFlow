from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from ml.training.sensor_metrics import (
    compute_binary_classification_metrics,
    compute_threshold_sweep,
    summarize_probability_distribution,
)
from ml.training.sensor_models import available_sensor_algorithms, build_sensor_estimator
from ml.training.sensor_multidomain import (
    balance_training_domains,
    class_balance,
    class_balance_by_domain,
    compute_domain_metrics,
    compute_macro_metrics,
    load_multidomain_sensor_feature_frame,
    row_count_by_domain,
    split_multidomain_by_session,
)
from ml.training.sensor_pipeline import FEATURE_COLUMNS
from ml.training.train_sensor import (
    _export_onnx_if_available,
    _predict_positive_probability,
    _save_sensor_model,
)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a domain-aware ChameleonFlow sensor on multiple datasets."
    )
    parser.add_argument("--inputs", nargs="+", required=True, type=Path, help="Input CSV/Parquet files")
    parser.add_argument(
        "--output-model",
        required=True,
        type=Path,
        help="Model output path",
    )
    parser.add_argument(
        "--output-metadata",
        required=True,
        type=Path,
        help="Training metadata JSON output path",
    )
    parser.add_argument(
        "--algorithm",
        choices=available_sensor_algorithms(),
        default="lightgbm",
        help="Estimator family to train",
    )
    parser.add_argument(
        "--output-onnx",
        type=Path,
        default=None,
        help="Optional ONNX output path",
    )
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--window-seconds", type=int, default=5)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--balance-domains",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Undersample larger domains to the smallest training-domain size",
    )
    return parser


def train_sensor_multidomain(
    *,
    input_paths: list[Path],
    output_model_path: Path,
    output_metadata_path: Path,
    output_onnx_path: Path | None,
    algorithm: str,
    threshold: float,
    window_seconds: int,
    validation_ratio: float,
    seed: int,
    balance_domains: bool,
) -> dict[str, Any]:
    feature_frame = load_multidomain_sensor_feature_frame(
        input_paths,
        window_seconds=window_seconds,
    )
    if feature_frame["label"].nunique() < 2:
        msg = "Multidomain sensor training requires at least two classes in the input labels."
        raise ValueError(msg)

    raw_train_frame, validation_frame = split_multidomain_by_session(
        feature_frame,
        validation_ratio=validation_ratio,
        seed=seed,
    )
    train_frame = (
        balance_training_domains(raw_train_frame, seed=seed)
        if balance_domains
        else raw_train_frame.reset_index(drop=True)
    )

    model = build_sensor_estimator(algorithm, seed=seed)
    training_started = time.perf_counter()
    model.fit(train_frame[FEATURE_COLUMNS], train_frame["label"])
    fit_seconds = time.perf_counter() - training_started
    model_format = _save_sensor_model(model, algorithm=algorithm, output_model_path=output_model_path)

    validation_probabilities = _predict_positive_probability(model, validation_frame)
    validation_metrics = compute_binary_classification_metrics(
        labels=validation_frame["label"].to_numpy(),
        probabilities=validation_probabilities,
        threshold=threshold,
    )
    threshold_sweep = compute_threshold_sweep(
        labels=validation_frame["label"].to_numpy(),
        probabilities=validation_probabilities,
    )
    probability_summary = summarize_probability_distribution(validation_probabilities)
    metrics_by_domain = compute_domain_metrics(
        frame=validation_frame,
        probabilities=validation_probabilities,
        threshold=threshold,
    )

    metadata = {
        "model_type": algorithm,
        "model_format": model_format,
        "task_semantics": "proxy_nonbaseline_traffic_across_domains",
        "feature_columns": FEATURE_COLUMNS,
        "threshold": threshold,
        "window_seconds": window_seconds,
        "seed": seed,
        "fit_seconds": fit_seconds,
        "domains": sorted(feature_frame["domain"].astype(str).unique().tolist()),
        "domain_notes": {
            str(domain): "dataset-native positive class preserved; interpret metrics as proxy anomaly/domain-shift scores"
            for domain in sorted(feature_frame["domain"].astype(str).unique().tolist())
        },
        "balance_domains": balance_domains,
        "input_paths": [str(input_path) for input_path in input_paths],
        "training_rows_before_balancing": int(len(raw_train_frame)),
        "training_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "training_rows_by_domain_before_balancing": row_count_by_domain(raw_train_frame),
        "training_rows_by_domain": row_count_by_domain(train_frame),
        "validation_rows_by_domain": row_count_by_domain(validation_frame),
        "class_balance": class_balance(train_frame),
        "class_balance_by_domain": class_balance_by_domain(train_frame),
        "validation_class_balance_by_domain": class_balance_by_domain(validation_frame),
        "validation_probability_summary": probability_summary,
        "validation_metrics": validation_metrics,
        "validation_metrics_by_domain": metrics_by_domain,
        "validation_metrics_macro": compute_macro_metrics(metrics_by_domain),
        "threshold_sweep": threshold_sweep,
    }

    if output_onnx_path is not None:
        if algorithm == "lightgbm":
            metadata["onnx_exported"] = _export_onnx_if_available(model, output_onnx_path)
        else:
            metadata["onnx_exported"] = False
            metadata["onnx_export_reason"] = "only_plain_lightgbm_is_supported"
    else:
        metadata["onnx_exported"] = False

    output_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    output_metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()
    metadata = train_sensor_multidomain(
        input_paths=args.inputs,
        output_model_path=args.output_model,
        output_metadata_path=args.output_metadata,
        output_onnx_path=args.output_onnx,
        algorithm=args.algorithm,
        threshold=args.threshold,
        window_seconds=args.window_seconds,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        balance_domains=args.balance_domains,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
