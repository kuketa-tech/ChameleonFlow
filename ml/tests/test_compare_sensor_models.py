from __future__ import annotations

import json

import pandas as pd

from ml.training.compare_sensor_models import compare_sensor_models


def test_compare_sensor_models_writes_ranked_summary(tmp_path) -> None:
    input_path = tmp_path / "sensor_metrics.csv"
    output_summary_path = tmp_path / "sensor_benchmark.json"

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

    summary = compare_sensor_models(
        input_path=input_path,
        output_summary_path=output_summary_path,
        algorithms=["lightgbm", "logistic_regression"],
        threshold=0.15,
        window_seconds=5,
        validation_ratio=0.25,
        seed=42,
        max_train_rows=None,
        max_validation_rows=None,
    )

    assert output_summary_path.exists()
    assert summary["best_algorithm"] in {"lightgbm", "logistic_regression"}
    assert len(summary["ranking"]) == 2

    restored = json.loads(output_summary_path.read_text(encoding="utf-8"))
    assert restored["ranking"][0]["threshold_sweep"]["best_by_f1"]["threshold"] >= 0.05
