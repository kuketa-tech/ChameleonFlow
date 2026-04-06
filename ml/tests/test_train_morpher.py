from __future__ import annotations

import json

import pandas as pd
import torch

from ml.training.train_morpher import train_morpher


def test_train_morpher_writes_checkpoint_and_metadata(tmp_path) -> None:
    input_path = tmp_path / "browser_iat.csv"
    output_model_path = tmp_path / "morpher.pt"
    output_metadata_path = tmp_path / "morpher.metadata.json"

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

    metadata = train_morpher(
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

    assert output_model_path.exists()
    assert output_metadata_path.exists()
    assert metadata["device"] == "cpu"
    assert metadata["training_examples"] > 0
    assert metadata["validation_metrics"] is not None
    assert "mae_ms" in metadata["validation_metrics"]

    checkpoint = torch.load(output_model_path, map_location="cpu")
    assert checkpoint["config"]["sequence_length"] == 5
    assert checkpoint["config"]["hidden_size"] == 8
    assert "state_dict" in checkpoint
    assert checkpoint["normalization"]["std"] > 0

    restored_metadata = json.loads(output_metadata_path.read_text(encoding="utf-8"))
    assert restored_metadata["validation_examples"] > 0
