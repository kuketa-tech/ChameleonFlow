from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.training.evaluate_sensor_transfer import evaluate_sensor_transfer


def test_evaluate_sensor_transfer_writes_summary(tmp_path: Path) -> None:
    train_input_path = tmp_path / "train.csv"
    eval_input_path = tmp_path / "eval.csv"
    output_summary_path = tmp_path / "transfer.json"

    train_rows: list[dict[str, int | float | str]] = []
    for session_id, label, packet_loss, rtt_ms, retransmissions, resets in [
        ("train-a", 0, 0, 20.0, 0, 0),
        ("train-b", 0, 1, 30.0, 0, 0),
        ("train-c", 1, 8, 180.0, 4, 1),
        ("train-d", 1, 7, 150.0, 3, 1),
    ]:
        for second in range(5):
            train_rows.append(
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

    eval_rows: list[dict[str, int | float | str]] = []
    for session_id, label, packet_loss, rtt_ms, retransmissions, resets in [
        ("eval-a", 0, 0, 22.0, 0, 0),
        ("eval-b", 1, 6, 170.0, 3, 1),
    ]:
        for second in range(5):
            eval_rows.append(
                {
                    "session_id": session_id,
                    "timestamp": f"2026-04-06T11:00:0{second}Z",
                    "packets_sent": 10,
                    "packets_lost": packet_loss,
                    "rtt_ms": rtt_ms + second,
                    "retransmissions": retransmissions,
                    "resets": resets,
                    "label": label,
                }
            )

    pd.DataFrame(train_rows).to_csv(train_input_path, index=False)
    pd.DataFrame(eval_rows).to_csv(eval_input_path, index=False)

    summary = evaluate_sensor_transfer(
        train_input_path=train_input_path,
        eval_input_path=eval_input_path,
        output_summary_path=output_summary_path,
        algorithm="lightgbm",
        threshold=0.15,
        window_seconds=5,
        seed=42,
        max_train_rows=None,
        max_eval_rows=None,
    )

    assert output_summary_path.exists()
    assert summary["algorithm"] == "lightgbm"
    assert summary["eval_metrics"]["f1"] >= 0.0
    assert summary["threshold_sweep"]["best_by_f1"]["threshold"] >= 0.05

    restored = json.loads(output_summary_path.read_text(encoding="utf-8"))
    assert restored["train_rows"] > 0
    assert restored["eval_rows"] > 0
