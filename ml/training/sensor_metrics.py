from __future__ import annotations

import math
from typing import Any

import numpy as np


def summarize_probability_distribution(probabilities: np.ndarray) -> dict[str, float]:
    if probabilities.size == 0:
        return {
            "min": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }

    return {
        "min": float(np.min(probabilities)),
        "p25": float(np.percentile(probabilities, 25)),
        "p50": float(np.percentile(probabilities, 50)),
        "p75": float(np.percentile(probabilities, 75)),
        "p95": float(np.percentile(probabilities, 95)),
        "max": float(np.max(probabilities)),
    }


def compute_binary_classification_metrics(
    *,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    predictions = (probabilities >= threshold).astype(int)
    metrics: dict[str, Any] = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "positive_rate": float(predictions.mean()) if len(predictions) else 0.0,
    }

    if np.unique(labels).size >= 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
        metrics["average_precision"] = float(average_precision_score(labels, probabilities))
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics["confusion_matrix"] = {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics


def compute_threshold_sweep(
    *,
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05)

    sweep = [
        compute_binary_classification_metrics(
            labels=labels,
            probabilities=probabilities,
            threshold=float(threshold),
        )
        for threshold in thresholds
    ]
    best = max(
        sweep,
        key=lambda metrics: (
            metrics["f1"],
            metrics["recall"],
            metrics["precision"],
            -metrics["threshold"],
        ),
    )
    return {
        "best_by_f1": best,
        "thresholds": sweep,
    }


def compute_regression_metrics(
    *,
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    rmse = math.sqrt(mean_squared_error(targets, predictions))
    return {
        "mae": float(mean_absolute_error(targets, predictions)),
        "rmse": float(rmse),
    }
