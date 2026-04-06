from __future__ import annotations

import json
from pathlib import Path

from ml.training.evaluate_sensor_loo import evaluate_sensor_loo
from ml.training.generate_controlled_sensor_domains import generate_controlled_sensor_domains


def test_evaluate_sensor_loo_writes_summary(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "domains"
    summary_path = tmp_path / "loo.json"
    generation = generate_controlled_sensor_domains(
        output_dir=dataset_dir,
        domains=["fiber_lab", "home_wifi", "satellite_emulated"],
        applications=["browsing", "streaming"],
        sessions_per_application=3,
        baseline_rows=5,
        impairment_rows=5,
        recovery_rows=3,
        seed=21,
    )

    summary = evaluate_sensor_loo(
        input_paths=[Path(path) for path in generation["generated_paths"]],
        output_summary_path=summary_path,
        algorithm="lightgbm",
        threshold=0.15,
        window_seconds=5,
        seed=21,
        balance_domains=True,
    )

    assert summary_path.exists()
    assert summary["algorithm"] == "lightgbm"
    assert set(summary["domains"]) == {"fiber_lab", "home_wifi", "satellite_emulated"}
    assert summary["macro_metrics"]["f1"] is not None
    assert summary["macro_best_by_f1"]["f1"] >= summary["macro_metrics"]["f1"]

    restored = json.loads(summary_path.read_text(encoding="utf-8"))
    assert set(restored["holdouts"]) == {"fiber_lab", "home_wifi", "satellite_emulated"}
