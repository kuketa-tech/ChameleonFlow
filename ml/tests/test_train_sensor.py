from __future__ import annotations

import json

import pandas as pd

from ml.training.train_sensor import train_sensor


def test_train_sensor_writes_model_and_validation_metrics(tmp_path) -> None:
    input_path = tmp_path / "sensor_metrics.csv"
    output_model_path = tmp_path / "sensor.txt"
    output_metadata_path = tmp_path / "sensor.metadata.json"

    rows: list[dict[str, int | float | str]] = []
    for session_id, label, packet_loss, rtt_ms, retransmissions, resets in [
        ("s1", 0, 0, 30.0, 0, 0),
        ("s2", 0, 1, 35.0, 0, 0),
        ("s3", 1, 8, 180.0, 3, 1),
        ("s4", 1, 7, 160.0, 2, 1),
    ]:
        for second in range(5):
            rows.append(
                {
                    "session_id": session_id,
                    "timestamp": f"2026-04-06T10:00:0{second}Z",
                    "packets_sent": 10,
                    "packets_lost": packet_loss,
                    "rtt_ms": rtt_ms + second,
                    "retransmissions": retransmissions,
                    "resets": resets,
                    "label": label,
                }
            )

    pd.DataFrame(rows).to_csv(input_path, index=False)

    metadata = train_sensor(
        input_path=input_path,
        output_model_path=output_model_path,
        output_metadata_path=output_metadata_path,
        output_onnx_path=None,
        algorithm="lightgbm",
        threshold=0.7,
        window_seconds=5,
        validation_ratio=0.25,
        seed=42,
    )

    assert output_model_path.exists()
    assert output_metadata_path.exists()
    assert metadata["training_rows"] > 0
    assert metadata["validation_rows"] > 0
    assert metadata["validation_metrics"] is not None
    assert "f1" in metadata["validation_metrics"]
    assert metadata["threshold_sweep"] is not None
    assert "best_by_f1" in metadata["threshold_sweep"]
    assert metadata["validation_probability_summary"] is not None

    restored_metadata = json.loads(output_metadata_path.read_text(encoding="utf-8"))
    assert restored_metadata["validation_metrics"]["threshold"] == 0.7


def test_train_sensor_supports_logistic_regression(tmp_path) -> None:
    input_path = tmp_path / "sensor_metrics.csv"
    output_model_path = tmp_path / "sensor.joblib"
    output_metadata_path = tmp_path / "sensor.metadata.json"

    rows: list[dict[str, int | float | str]] = []
    for session_id, label, packet_loss, rtt_ms, retransmissions, resets in [
        ("s1", 0, 0, 30.0, 0, 0),
        ("s2", 0, 1, 35.0, 0, 0),
        ("s3", 1, 8, 180.0, 3, 1),
        ("s4", 1, 7, 160.0, 2, 1),
    ]:
        for second in range(5):
            rows.append(
                {
                    "session_id": session_id,
                    "timestamp": f"2026-04-06T10:00:0{second}Z",
                    "packets_sent": 10,
                    "packets_lost": packet_loss,
                    "rtt_ms": rtt_ms + second,
                    "retransmissions": retransmissions,
                    "resets": resets,
                    "label": label,
                }
            )

    pd.DataFrame(rows).to_csv(input_path, index=False)

    metadata = train_sensor(
        input_path=input_path,
        output_model_path=output_model_path,
        output_metadata_path=output_metadata_path,
        output_onnx_path=None,
        algorithm="logistic_regression",
        threshold=0.15,
        window_seconds=5,
        validation_ratio=0.25,
        seed=42,
    )

    assert output_model_path.exists()
    assert metadata["model_type"] == "logistic_regression"
    assert metadata["model_format"] == "joblib"
