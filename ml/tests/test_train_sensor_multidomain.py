from __future__ import annotations

import json

import pandas as pd

from ml.training.train_sensor_multidomain import train_sensor_multidomain


def test_train_sensor_multidomain_writes_domain_metrics(tmp_path) -> None:
    domain_a_path = tmp_path / "domain_a.csv"
    domain_b_path = tmp_path / "domain_b.csv"
    output_model_path = tmp_path / "sensor.txt"
    output_metadata_path = tmp_path / "sensor.metadata.json"

    domain_a_rows: list[dict[str, int | float | str]] = []
    for session_id, label, packet_loss, rtt_ms, retransmissions, resets in [
        ("a1", 0, 0, 20.0, 0, 0),
        ("a2", 0, 1, 25.0, 0, 0),
        ("a3", 1, 7, 160.0, 3, 1),
        ("a4", 1, 6, 150.0, 2, 1),
    ]:
        for second in range(5):
            domain_a_rows.append(
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

    domain_b_rows: list[dict[str, int | float | str]] = []
    for session_id, label, packet_loss, rtt_ms, retransmissions, resets in [
        ("b1", 0, 0, 22.0, 0, 0),
        ("b2", 0, 1, 28.0, 0, 0),
        ("b3", 1, 8, 180.0, 4, 1),
        ("b4", 1, 7, 170.0, 3, 1),
    ]:
        for second in range(5):
            domain_b_rows.append(
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

    pd.DataFrame(domain_a_rows).to_csv(domain_a_path, index=False)
    pd.DataFrame(domain_b_rows).to_csv(domain_b_path, index=False)

    metadata = train_sensor_multidomain(
        input_paths=[domain_a_path, domain_b_path],
        output_model_path=output_model_path,
        output_metadata_path=output_metadata_path,
        output_onnx_path=None,
        algorithm="lightgbm",
        threshold=0.15,
        window_seconds=5,
        validation_ratio=0.25,
        seed=42,
        balance_domains=True,
    )

    assert output_model_path.exists()
    assert output_metadata_path.exists()
    assert metadata["task_semantics"] == "proxy_nonbaseline_traffic_across_domains"
    assert sorted(metadata["domains"]) == ["domain_a", "domain_b"]
    assert metadata["validation_metrics_macro"]["f1"] is not None
    assert set(metadata["validation_metrics_by_domain"]) == {"domain_a", "domain_b"}

    restored_metadata = json.loads(output_metadata_path.read_text(encoding="utf-8"))
    assert restored_metadata["balance_domains"] is True
