from __future__ import annotations

import pandas as pd

from ml.training.infer_morpher import infer_morpher
from ml.training.train_morpher import train_morpher


def test_infer_morpher_writes_prediction_rows(tmp_path) -> None:
    input_path = tmp_path / "browser_iat.csv"
    output_model_path = tmp_path / "morpher.pt"
    output_metadata_path = tmp_path / "morpher.metadata.json"
    output_predictions_path = tmp_path / "morpher_predictions.csv"

    rows: list[dict[str, float | int | str]] = []
    for trace_id, offset in [("trace-a", 0.0), ("trace-b", 5.0)]:
        for packet_index in range(30):
            rows.append(
                {
                    "trace_id": trace_id,
                    "packet_index": packet_index,
                    "iat_ms": 20.0 + offset + packet_index * 0.5,
                }
            )

    pd.DataFrame(rows).to_csv(input_path, index=False)

    train_morpher(
        input_path=input_path,
        output_model_path=output_model_path,
        output_metadata_path=output_metadata_path,
        output_onnx_path=None,
        sequence_length=5,
        hidden_size=8,
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
        validation_ratio=0.2,
        seed=42,
        requested_device="cpu",
    )

    summary = infer_morpher(
        input_path=input_path,
        model_path=output_model_path,
        output_path=output_predictions_path,
        requested_device="cpu",
    )

    assert output_predictions_path.exists()
    prediction_frame = pd.read_csv(output_predictions_path)
    assert summary["rows"] == len(prediction_frame)
    assert {"trace_id", "target_packet_index", "actual_iat_ms", "predicted_iat_ms", "abs_error_ms"} <= set(
        prediction_frame.columns
    )
