from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from ml.training.sensor_metrics import (
    compute_binary_classification_metrics,
    compute_threshold_sweep,
    summarize_probability_distribution,
)
from ml.training.sensor_models import available_sensor_algorithms, build_sensor_estimator
from ml.training.sensor_pipeline import FEATURE_COLUMNS, build_sensor_feature_frame, load_table
from ml.training.train_sensor import _predict_positive_probability


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train sensor on one dataset and evaluate on another.")
    parser.add_argument("--train-input", required=True, type=Path, help="Training CSV/Parquet")
    parser.add_argument("--eval-input", required=True, type=Path, help="Evaluation CSV/Parquet")
    parser.add_argument("--output-summary", required=True, type=Path, help="JSON output path")
    parser.add_argument(
        "--algorithm",
        choices=available_sensor_algorithms(),
        default="lightgbm",
        help="Estimator family to train",
    )
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--window-seconds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-eval-rows", type=int, default=None)
    return parser


def _downsample_frame(frame: pd.DataFrame, *, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or len(frame) <= max_rows:
        return frame.reset_index(drop=True)

    return frame.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def _class_balance(frame: pd.DataFrame) -> dict[str, int]:
    return {
        str(label): int(count)
        for label, count in frame["label"].value_counts().sort_index().items()
    }


def evaluate_sensor_transfer(
    *,
    train_input_path: Path,
    eval_input_path: Path,
    output_summary_path: Path,
    algorithm: str,
    threshold: float,
    window_seconds: int,
    seed: int,
    max_train_rows: int | None,
    max_eval_rows: int | None,
) -> dict[str, Any]:
    train_raw = load_table(train_input_path)
    eval_raw = load_table(eval_input_path)

    train_frame = build_sensor_feature_frame(train_raw, window_seconds=window_seconds)
    eval_frame = build_sensor_feature_frame(eval_raw, window_seconds=window_seconds)

    train_frame = _downsample_frame(train_frame, max_rows=max_train_rows, seed=seed)
    eval_frame = _downsample_frame(eval_frame, max_rows=max_eval_rows, seed=seed)

    if train_frame["label"].nunique() < 2:
        msg = "Transfer training requires at least two classes in the training dataset."
        raise ValueError(msg)

    estimator = build_sensor_estimator(algorithm, seed=seed)
    started = time.perf_counter()
    estimator.fit(train_frame[FEATURE_COLUMNS], train_frame["label"])
    fit_seconds = time.perf_counter() - started

    eval_probabilities = _predict_positive_probability(estimator, eval_frame)
    eval_labels = eval_frame["label"].to_numpy()
    eval_metrics = compute_binary_classification_metrics(
        labels=eval_labels,
        probabilities=eval_probabilities,
        threshold=threshold,
    )
    threshold_sweep = compute_threshold_sweep(
        labels=eval_labels,
        probabilities=eval_probabilities,
    )

    summary = {
        "algorithm": algorithm,
        "threshold": threshold,
        "window_seconds": window_seconds,
        "seed": seed,
        "fit_seconds": fit_seconds,
        "feature_columns": FEATURE_COLUMNS,
        "train_input": str(train_input_path),
        "eval_input": str(eval_input_path),
        "train_rows": int(len(train_frame)),
        "eval_rows": int(len(eval_frame)),
        "train_class_balance": _class_balance(train_frame),
        "eval_class_balance": _class_balance(eval_frame),
        "eval_metrics": eval_metrics,
        "threshold_sweep": threshold_sweep,
        "eval_probability_summary": summarize_probability_distribution(eval_probabilities),
    }

    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()
    summary = evaluate_sensor_transfer(
        train_input_path=args.train_input,
        eval_input_path=args.eval_input,
        output_summary_path=args.output_summary,
        algorithm=args.algorithm,
        threshold=args.threshold,
        window_seconds=args.window_seconds,
        seed=args.seed,
        max_train_rows=args.max_train_rows,
        max_eval_rows=args.max_eval_rows,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
