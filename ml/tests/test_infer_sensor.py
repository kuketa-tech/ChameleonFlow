from __future__ import annotations

import pandas as pd

from ml.training.infer_sensor import infer_sensor
from ml.training.train_sensor import train_sensor


def test_infer_sensor_writes_prediction_rows(tmp_path) -> None:
    input_path = tmp_path / "sensor_metrics.csv"
    output_model_path = tmp_path / "sensor.txt"
    output_metadata_path = tmp_path / "sensor.metadata.json"
    output_predictions_path = tmp_path / "sensor_predictions.csv"

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

    train_sensor(
        input_path=input_path,
        output_model_path=output_model_path,
        output_metadata_path=output_metadata_path,
        output_onnx_path=None,
        algorithm="lightgbm",
        threshold=0.15,
        window_seconds=5,
        validation_ratio=0.25,
        seed=42,
    )

    summary = infer_sensor(
        input_path=input_path,
        model_path=output_model_path,
        output_path=output_predictions_path,
        metadata_path=output_metadata_path,
        threshold=None,
        window_seconds=5,
    )

    assert output_predictions_path.exists()
    prediction_frame = pd.read_csv(output_predictions_path)
    assert summary["rows"] == len(prediction_frame)
    assert {"session_id", "window_start", "label", "probability", "degraded"} <= set(
        prediction_frame.columns
    )
