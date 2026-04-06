from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ml.training.sensor_pipeline import FEATURE_COLUMNS, build_sensor_feature_frame, load_table


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ChameleonFlow sensor inference on raw metrics.")
    parser.add_argument("--input", required=True, type=Path, help="Raw metrics CSV/Parquet")
    parser.add_argument("--model", required=True, type=Path, help="Sensor model path")
    parser.add_argument("--output", required=True, type=Path, help="Predictions CSV/Parquet")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional training metadata JSON to read threshold/model format from",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--window-seconds", type=int, default=5)
    return parser


def load_sensor_model(model_path: Path, *, metadata_path: Path | None) -> tuple[Any, str, float | None]:
    model_format = "joblib"
    threshold: float | None = None

    if metadata_path is not None:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model_format = metadata.get("model_format", model_format)
        threshold_value = metadata.get("threshold")
        threshold = float(threshold_value) if threshold_value is not None else None
    elif model_path.suffix == ".txt":
        model_format = "lightgbm_booster"

    if model_format == "lightgbm_booster":
        from lightgbm import Booster

        return Booster(model_file=str(model_path)), model_format, threshold

    return joblib.load(model_path), model_format, threshold


def _predict_positive_probability(model: Any, *, model_format: str, feature_frame: pd.DataFrame) -> Any:
    if model_format == "lightgbm_booster":
        return model.predict(feature_frame[FEATURE_COLUMNS])

    return model.predict_proba(feature_frame[FEATURE_COLUMNS])[:, 1]


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


def infer_sensor(
    *,
    input_path: Path,
    model_path: Path,
    output_path: Path,
    metadata_path: Path | None,
    threshold: float | None,
    window_seconds: int,
) -> dict[str, Any]:
    raw_frame = load_table(input_path)
    feature_frame = build_sensor_feature_frame(raw_frame, window_seconds=window_seconds)
    model, model_format, metadata_threshold = load_sensor_model(model_path, metadata_path=metadata_path)
    effective_threshold = threshold if threshold is not None else metadata_threshold
    if effective_threshold is None:
        effective_threshold = 0.15

    probabilities = _predict_positive_probability(
        model,
        model_format=model_format,
        feature_frame=feature_frame,
    )
    prediction_frame = feature_frame[["session_id", "window_start", "label"]].copy()
    prediction_frame["probability"] = probabilities
    prediction_frame["degraded"] = prediction_frame["probability"] >= float(effective_threshold)
    save_prediction_frame(prediction_frame, output_path)

    return {
        "rows": int(len(prediction_frame)),
        "threshold": float(effective_threshold),
        "model_format": model_format,
    }


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()
    summary = infer_sensor(
        input_path=args.input,
        model_path=args.model,
        output_path=args.output,
        metadata_path=args.metadata,
        threshold=args.threshold,
        window_seconds=args.window_seconds,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
