from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.training.sensor_metrics import (
    compute_binary_classification_metrics,
    compute_threshold_sweep,
    summarize_probability_distribution,
)
from ml.training.sensor_pipeline import build_sensor_feature_frame, load_table

MACRO_METRIC_KEYS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "positive_rate",
    "roc_auc",
    "average_precision",
)


def load_multidomain_sensor_feature_frame(
    input_paths: Sequence[Path],
    *,
    window_seconds: int,
    domain_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    if not input_paths:
        msg = "At least one input path is required."
        raise ValueError(msg)

    if domain_names is None:
        domain_names = [input_path.stem for input_path in input_paths]

    if len(domain_names) != len(input_paths):
        msg = "domain_names must have the same length as input_paths."
        raise ValueError(msg)

    frames: list[pd.DataFrame] = []
    for input_path, domain_name in zip(input_paths, domain_names, strict=True):
        raw_frame = load_table(input_path)
        feature_frame = build_sensor_feature_frame(raw_frame, window_seconds=window_seconds).copy()
        feature_frame["domain"] = str(domain_name)
        feature_frame["session_id"] = (
            feature_frame["domain"].astype(str) + "::" + feature_frame["session_id"].astype(str)
        )
        frames.append(feature_frame)

    return pd.concat(frames, ignore_index=True)


def split_multidomain_by_session(
    feature_frame: pd.DataFrame,
    *,
    validation_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    if not 0.0 <= validation_ratio < 1.0:
        msg = "validation_ratio must be in the range [0.0, 1.0)."
        raise ValueError(msg)

    train_frames: list[pd.DataFrame] = []
    validation_frames: list[pd.DataFrame] = []

    for _, domain_frame in feature_frame.groupby("domain", sort=True):
        session_labels = (
            domain_frame.groupby("session_id", as_index=False)["label"].max().assign(
                session_id=lambda frame: frame["session_id"].astype(str)
            )
        )
        unique_sessions: Sequence[str] = session_labels["session_id"].tolist()
        if len(unique_sessions) < 2 or validation_ratio == 0.0:
            train_frames.append(domain_frame.reset_index(drop=True))
            continue

        validation_size = max(1, int(round(len(unique_sessions) * validation_ratio)))
        stratify = None
        if (
            session_labels["label"].nunique() >= 2
            and validation_size >= session_labels["label"].nunique()
        ):
            stratify = session_labels["label"]

        train_sessions, validation_sessions = train_test_split(
            unique_sessions,
            test_size=validation_ratio,
            random_state=seed,
            stratify=stratify,
        )
        train_frames.append(
            domain_frame[domain_frame["session_id"].isin(train_sessions)].reset_index(drop=True)
        )
        validation_frames.append(
            domain_frame[domain_frame["session_id"].isin(validation_sessions)].reset_index(drop=True)
        )

    train_frame = pd.concat(train_frames, ignore_index=True) if train_frames else feature_frame.iloc[0:0].copy()
    validation_frame = (
        pd.concat(validation_frames, ignore_index=True)
        if validation_frames
        else feature_frame.iloc[0:0].copy()
    )
    if validation_ratio > 0.0 and validation_frame.empty:
        msg = "validation split produced an empty validation frame."
        raise ValueError(msg)

    return train_frame, validation_frame


def balance_training_domains(train_frame: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    domain_counts = train_frame["domain"].value_counts()
    if len(domain_counts) < 2:
        return train_frame.reset_index(drop=True)

    target_rows = int(domain_counts.min())
    balanced_frames: list[pd.DataFrame] = []
    for _, domain_frame in train_frame.groupby("domain", sort=True):
        if len(domain_frame) > target_rows:
            balanced_frames.append(domain_frame.sample(n=target_rows, random_state=seed))
        else:
            balanced_frames.append(domain_frame)

    return pd.concat(balanced_frames, ignore_index=True).sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)


def class_balance(frame: pd.DataFrame) -> dict[str, int]:
    return {
        str(label): int(count)
        for label, count in frame["label"].value_counts().sort_index().items()
    }


def row_count_by_domain(frame: pd.DataFrame) -> dict[str, int]:
    return {
        str(domain): int(count)
        for domain, count in frame["domain"].value_counts().sort_index().items()
    }


def class_balance_by_domain(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    balances: dict[str, dict[str, int]] = {}
    for domain, domain_frame in frame.groupby("domain", sort=True):
        balances[str(domain)] = class_balance(domain_frame)
    return balances


def compute_domain_metrics(
    *,
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, dict[str, Any]]:
    by_domain: dict[str, dict[str, Any]] = {}
    for domain, domain_frame in frame.groupby("domain", sort=True):
        mask = frame["domain"].eq(domain).to_numpy()
        domain_probabilities = probabilities[mask]
        domain_labels = domain_frame["label"].to_numpy()
        metrics = compute_binary_classification_metrics(
            labels=domain_labels,
            probabilities=domain_probabilities,
            threshold=threshold,
        )
        by_domain[str(domain)] = {
            "rows": int(len(domain_frame)),
            "class_balance": class_balance(domain_frame),
            "metrics": metrics,
            "threshold_sweep": compute_threshold_sweep(
                labels=domain_labels,
                probabilities=domain_probabilities,
            ),
            "probability_summary": summarize_probability_distribution(domain_probabilities),
        }
    return by_domain


def compute_macro_metrics(metrics_by_domain: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    macro_metrics: dict[str, float | None] = {}
    domain_metrics = [entry["metrics"] for entry in metrics_by_domain.values()]
    for key in MACRO_METRIC_KEYS:
        values = [metric[key] for metric in domain_metrics if metric.get(key) is not None]
        macro_metrics[key] = float(np.mean(values)) if values else None
    return macro_metrics
