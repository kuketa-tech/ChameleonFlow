from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import onnx
import pandas as pd

from ml.training.sensor_metrics import (
    compute_binary_classification_metrics,
    compute_threshold_sweep,
    summarize_probability_distribution,
)
from ml.training.sensor_models import (
    available_sensor_algorithms,
    build_sensor_estimator,
    get_sensor_model_spec,
)
from ml.training.sensor_pipeline import FEATURE_COLUMNS, build_sensor_feature_frame, load_table


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the ChameleonFlow channel sensor.")
    parser.add_argument("--input", required=True, type=Path, help="Raw metrics CSV/Parquet file")
    parser.add_argument("--output-model", required=True, type=Path, help="LightGBM model output path")
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
    return parser


def _export_onnx_if_available(model: Any, output_path: Path) -> bool:
    try:
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType
    except ImportError:
        return False

    onnx_model = onnxmltools.convert_lightgbm(
        model,
        initial_types=[("input", FloatTensorType([None, len(FEATURE_COLUMNS)]))],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(onnx_model, output_path)
    return True


def _predict_positive_probability(model: Any, frame: pd.DataFrame) -> Any:
    return model.predict_proba(frame[FEATURE_COLUMNS])[:, 1]


def _save_sensor_model(model: Any, *, algorithm: str, output_model_path: Path) -> str:
    spec = get_sensor_model_spec(algorithm)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    if spec.serializer == "lightgbm_booster":
        model.booster_.save_model(str(output_model_path))
        return "lightgbm_booster"

    joblib.dump(model, output_model_path)
    return "joblib"


def _split_by_session(
    feature_frame: pd.DataFrame,
    *,
    validation_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    if not 0.0 <= validation_ratio < 1.0:
        msg = "validation_ratio must be in the range [0.0, 1.0)."
        raise ValueError(msg)

    session_labels = (
        feature_frame.groupby("session_id", as_index=False)["label"].max().assign(
            session_id=lambda frame: frame["session_id"].astype(str)
        )
    )
    unique_sessions: Sequence[str] = session_labels["session_id"].tolist()
    if len(unique_sessions) < 2 or validation_ratio == 0.0:
        return feature_frame.reset_index(drop=True), feature_frame.iloc[0:0].copy()

    validation_size = max(1, int(round(len(unique_sessions) * validation_ratio)))
    stratify = None
    if session_labels["label"].nunique() >= 2 and validation_size >= session_labels["label"].nunique():
        stratify = session_labels["label"]

    train_sessions, validation_sessions = train_test_split(
        unique_sessions,
        test_size=validation_ratio,
        random_state=seed,
        stratify=stratify,
    )
    train_frame = feature_frame[feature_frame["session_id"].isin(train_sessions)].reset_index(drop=True)
    validation_frame = feature_frame[
        feature_frame["session_id"].isin(validation_sessions)
    ].reset_index(drop=True)
    if validation_frame.empty:
        msg = "validation split produced an empty validation frame."
        raise ValueError(msg)

    return train_frame, validation_frame


def train_sensor(
    *,
    input_path: Path,
    output_model_path: Path,
    output_metadata_path: Path,
    output_onnx_path: Path | None,
    algorithm: str,
    threshold: float,
    window_seconds: int,
    validation_ratio: float,
    seed: int,
) -> dict[str, Any]:
    raw_frame = load_table(input_path)
    feature_frame = build_sensor_feature_frame(raw_frame, window_seconds=window_seconds)

    if feature_frame["label"].nunique() < 2:
        msg = "Sensor training requires at least two classes in the input labels."
        raise ValueError(msg)

    train_frame, validation_frame = _split_by_session(
        feature_frame,
        validation_ratio=validation_ratio,
        seed=seed,
    )

    model = build_sensor_estimator(algorithm, seed=seed)
    training_started = time.perf_counter()
    model.fit(train_frame[FEATURE_COLUMNS], train_frame["label"])
    fit_seconds = time.perf_counter() - training_started
    model_format = _save_sensor_model(model, algorithm=algorithm, output_model_path=output_model_path)

    validation_metrics: dict[str, Any] | None = None
    threshold_sweep: dict[str, Any] | None = None
    probability_summary: dict[str, float] | None = None
    if not validation_frame.empty:
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

    metadata = {
        "model_type": algorithm,
        "model_format": model_format,
        "feature_columns": FEATURE_COLUMNS,
        "threshold": threshold,
        "window_seconds": window_seconds,
        "seed": seed,
        "fit_seconds": fit_seconds,
        "training_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "class_balance": {
            str(label): int(count)
            for label, count in train_frame["label"].value_counts().sort_index().items()
        },
        "validation_probability_summary": probability_summary,
        "validation_metrics": validation_metrics,
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
    metadata = train_sensor(
        input_path=args.input,
        output_model_path=args.output_model,
        output_metadata_path=args.output_metadata,
        output_onnx_path=args.output_onnx,
        algorithm=args.algorithm,
        threshold=args.threshold,
        window_seconds=args.window_seconds,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
