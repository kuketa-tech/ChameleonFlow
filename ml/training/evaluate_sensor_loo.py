from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.training.sensor_metrics import (
    compute_binary_classification_metrics,
    compute_threshold_sweep,
    summarize_probability_distribution,
)
from ml.training.sensor_models import build_sensor_estimator
from ml.training.sensor_multidomain import (
    MACRO_METRIC_KEYS,
    balance_training_domains,
    class_balance,
)
from ml.training.sensor_pipeline import FEATURE_COLUMNS, build_sensor_feature_frame, load_table
from ml.training.train_sensor import _predict_positive_probability


def _load_domain_frames(
    input_paths: list[Path],
    *,
    window_seconds: int,
) -> dict[str, pd.DataFrame]:
    if len(input_paths) < 2:
        msg = "Leave-one-domain-out evaluation requires at least two input datasets."
        raise ValueError(msg)

    domain_frames: dict[str, pd.DataFrame] = {}
    for input_path in input_paths:
        domain_name = input_path.stem
        if domain_name in domain_frames:
            msg = f"Duplicate domain name derived from input path: {domain_name}"
            raise ValueError(msg)
        raw_frame = load_table(input_path)
        feature_frame = build_sensor_feature_frame(raw_frame, window_seconds=window_seconds).copy()
        feature_frame["domain"] = domain_name
        feature_frame["session_id"] = (
            feature_frame["domain"].astype(str) + "::" + feature_frame["session_id"].astype(str)
        )
        if feature_frame["label"].nunique() < 2:
            msg = f"Domain {domain_name} must contain at least two classes for LOO evaluation."
            raise ValueError(msg)
        domain_frames[domain_name] = feature_frame
    return domain_frames


def _mean_from_metrics(metrics_list: list[dict[str, Any]]) -> dict[str, float | None]:
    summary: dict[str, float | None] = {}
    for key in MACRO_METRIC_KEYS:
        values = [metrics[key] for metrics in metrics_list if metrics.get(key) is not None]
        summary[key] = float(np.mean(values)) if values else None
    return summary


def _mean_best_threshold_metrics(entries: list[dict[str, Any]]) -> dict[str, float | None]:
    keys = ("f1", "precision", "recall", "positive_rate", "roc_auc", "average_precision")
    summary: dict[str, float | None] = {}
    for key in keys:
        values = [entry[key] for entry in entries if entry.get(key) is not None]
        summary[key] = float(np.mean(values)) if values else None
    summary["threshold"] = float(np.mean([entry["threshold"] for entry in entries]))
    return summary


def evaluate_sensor_loo(
    *,
    input_paths: list[Path],
    output_summary_path: Path,
    algorithm: str,
    threshold: float,
    window_seconds: int,
    seed: int,
    balance_domains: bool,
) -> dict[str, Any]:
    domain_frames = _load_domain_frames(input_paths, window_seconds=window_seconds)
    holdouts: dict[str, dict[str, Any]] = {}
    default_metric_entries: list[dict[str, Any]] = []
    best_metric_entries: list[dict[str, Any]] = []

    for holdout_domain, eval_frame in sorted(domain_frames.items()):
        train_frames = [
            frame.copy()
            for domain_name, frame in domain_frames.items()
            if domain_name != holdout_domain
        ]
        train_frame = pd.concat(train_frames, ignore_index=True)
        if balance_domains:
            train_frame = balance_training_domains(train_frame, seed=seed)

        estimator = build_sensor_estimator(algorithm, seed=seed)
        started = time.perf_counter()
        estimator.fit(train_frame[FEATURE_COLUMNS], train_frame["label"])
        fit_seconds = time.perf_counter() - started

        probabilities = _predict_positive_probability(estimator, eval_frame)
        metrics = compute_binary_classification_metrics(
            labels=eval_frame["label"].to_numpy(),
            probabilities=probabilities,
            threshold=threshold,
        )
        threshold_sweep = compute_threshold_sweep(
            labels=eval_frame["label"].to_numpy(),
            probabilities=probabilities,
        )
        best_by_f1 = threshold_sweep["best_by_f1"]
        default_metric_entries.append(metrics)
        best_metric_entries.append(best_by_f1)

        holdouts[holdout_domain] = {
            "fit_seconds": fit_seconds,
            "train_rows": int(len(train_frame)),
            "eval_rows": int(len(eval_frame)),
            "train_domains": sorted(
                train_frame["domain"].astype(str).unique().tolist()
            ),
            "train_rows_by_domain": {
                str(domain_name): int(count)
                for domain_name, count in train_frame["domain"].value_counts().sort_index().items()
            },
            "train_class_balance": class_balance(train_frame),
            "eval_class_balance": class_balance(eval_frame),
            "metrics": metrics,
            "threshold_sweep": threshold_sweep,
            "probability_summary": summarize_probability_distribution(probabilities),
        }

    summary = {
        "algorithm": algorithm,
        "feature_columns": FEATURE_COLUMNS,
        "threshold": threshold,
        "window_seconds": window_seconds,
        "seed": seed,
        "balance_domains": balance_domains,
        "domains": sorted(domain_frames),
        "macro_metrics": _mean_from_metrics(default_metric_entries),
        "macro_best_by_f1": _mean_best_threshold_metrics(best_metric_entries),
        "holdouts": holdouts,
    }
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
