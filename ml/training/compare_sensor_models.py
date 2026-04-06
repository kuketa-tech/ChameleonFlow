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
from ml.training.train_sensor import _predict_positive_probability, _split_by_session


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare multiple sensor estimators on one split.")
    parser.add_argument("--input", required=True, type=Path, help="Raw metrics CSV/Parquet file")
    parser.add_argument(
        "--output-summary",
        required=True,
        type=Path,
        help="Benchmark summary JSON output path",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=available_sensor_algorithms(),
        default=["lightgbm", "lightgbm_sigmoid", "hist_gradient_boosting", "extra_trees", "random_forest", "logistic_regression"],
        help="Algorithms to benchmark",
    )
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--window-seconds", type=int, default=5)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-validation-rows", type=int, default=None)
    return parser


def _downsample_frame(frame: pd.DataFrame, *, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or len(frame) <= max_rows:
        return frame

    return frame.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def compare_sensor_models(
    *,
    input_path: Path,
    output_summary_path: Path,
    algorithms: list[str],
    threshold: float,
    window_seconds: int,
    validation_ratio: float,
    seed: int,
    max_train_rows: int | None,
    max_validation_rows: int | None,
) -> dict[str, Any]:
    raw_frame = load_table(input_path)
    feature_frame = build_sensor_feature_frame(raw_frame, window_seconds=window_seconds)
    train_frame, validation_frame = _split_by_session(
        feature_frame,
        validation_ratio=validation_ratio,
        seed=seed,
    )
    train_frame = _downsample_frame(train_frame, max_rows=max_train_rows, seed=seed)
    validation_frame = _downsample_frame(validation_frame, max_rows=max_validation_rows, seed=seed)

    labels = validation_frame["label"].to_numpy()
    results: list[dict[str, Any]] = []

    for algorithm in algorithms:
        estimator = build_sensor_estimator(algorithm, seed=seed)
        started = time.perf_counter()
        estimator.fit(train_frame[FEATURE_COLUMNS], train_frame["label"])
        fit_seconds = time.perf_counter() - started
        probabilities = _predict_positive_probability(estimator, validation_frame)
        validation_metrics = compute_binary_classification_metrics(
            labels=labels,
            probabilities=probabilities,
            threshold=threshold,
        )
        threshold_sweep = compute_threshold_sweep(
            labels=labels,
            probabilities=probabilities,
        )
        results.append(
            {
                "algorithm": algorithm,
                "fit_seconds": fit_seconds,
                "validation_metrics": validation_metrics,
                "threshold_sweep": threshold_sweep,
                "validation_probability_summary": summarize_probability_distribution(probabilities),
            }
        )

    ranking = sorted(
        results,
        key=lambda result: (
            result["threshold_sweep"]["best_by_f1"]["f1"],
            result["threshold_sweep"]["best_by_f1"]["recall"],
            result["validation_metrics"]["average_precision"] or 0.0,
            result["validation_metrics"]["roc_auc"] or 0.0,
        ),
        reverse=True,
    )

    summary = {
        "window_seconds": window_seconds,
        "threshold": threshold,
        "seed": seed,
        "training_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "ranking": ranking,
        "best_algorithm": ranking[0]["algorithm"] if ranking else None,
    }

    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()
    summary = compare_sensor_models(
        input_path=args.input,
        output_summary_path=args.output_summary,
        algorithms=args.algorithms,
        threshold=args.threshold,
        window_seconds=args.window_seconds,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        max_train_rows=args.max_train_rows,
        max_validation_rows=args.max_validation_rows,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
