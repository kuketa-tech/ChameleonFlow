from __future__ import annotations

import json

import pandas as pd

from ml.training.sensor_experiment_dataset import (
    build_sensor_experiment_dataset,
    initialize_sensor_experiment_run,
    label_sensor_experiment_frame,
    load_sensor_experiment_manifest,
)


def test_initialize_sensor_experiment_run_creates_manifest_and_template(tmp_path) -> None:
    run_dir = tmp_path / "run-001"
    manifest = initialize_sensor_experiment_run(
        run_dir=run_dir,
        run_id="run-001",
        domain="lab_wifi",
        application="browsing",
        scenario="delay-jitter",
    )

    assert manifest.run_id == "run-001"
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "sensor_metrics_raw.csv").exists()

    restored = load_sensor_experiment_manifest(run_dir / "manifest.json")
    assert restored.domain == "lab_wifi"
    assert len(restored.phases) == 3


def test_label_sensor_experiment_frame_assigns_phase_labels() -> None:
    manifest = load_sensor_experiment_manifest.__globals__["SensorExperimentManifest"].model_validate(
        {
            "schema_version": 1,
            "run_id": "run-002",
            "created_at": "2026-04-07T10:00:00Z",
            "domain": "home_ethernet",
            "application": "streaming",
            "scenario": "loss-burst",
            "window_seconds": 5,
            "files": {"raw_metrics": "sensor_metrics_raw.csv"},
            "phases": [
                {
                    "name": "baseline",
                    "start_offset_seconds": 0.0,
                    "duration_seconds": 10.0,
                    "label": 0,
                    "impairment_type": "baseline",
                    "severity": "none",
                },
                {
                    "name": "loss",
                    "start_offset_seconds": 10.0,
                    "duration_seconds": 10.0,
                    "label": 1,
                    "impairment_type": "loss",
                    "severity": "moderate",
                },
            ],
        }
    )
    raw_frame = pd.DataFrame(
        {
            "session_id": ["s1", "s1", "s1", "s1"],
            "timestamp": [
                "2026-04-07T10:00:00Z",
                "2026-04-07T10:00:05Z",
                "2026-04-07T10:00:10Z",
                "2026-04-07T10:00:15Z",
            ],
            "packets_sent": [10, 10, 10, 10],
            "packets_lost": [0, 0, 2, 2],
            "rtt_ms": [20.0, 21.0, 80.0, 90.0],
            "retransmissions": [0, 0, 1, 2],
            "resets": [0, 0, 0, 0],
        }
    )

    labeled = label_sensor_experiment_frame(raw_frame=raw_frame, manifest=manifest)

    assert labeled["label"].tolist() == [0, 0, 1, 1]
    assert labeled["phase_name"].tolist() == ["baseline", "baseline", "loss", "loss"]
    assert labeled["run_id"].nunique() == 1
    assert labeled["domain"].iloc[0] == "home_ethernet"


def test_build_sensor_experiment_dataset_combines_runs(tmp_path) -> None:
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    initialize_sensor_experiment_run(
        run_dir=run_a,
        run_id="run-a",
        domain="lab",
        application="browsing",
        scenario="delay",
        baseline_seconds=5.0,
        impairment_seconds=5.0,
        recovery_seconds=5.0,
    )
    initialize_sensor_experiment_run(
        run_dir=run_b,
        run_id="run-b",
        domain="home",
        application="streaming",
        scenario="loss",
        baseline_seconds=5.0,
        impairment_seconds=5.0,
        recovery_seconds=5.0,
    )

    for run_dir in [run_a, run_b]:
        pd.DataFrame(
            {
                "session_id": ["s1", "s1", "s1"],
                "timestamp": [
                    "2026-04-07T10:00:00Z",
                    "2026-04-07T10:00:06Z",
                    "2026-04-07T10:00:12Z",
                ],
                "packets_sent": [10, 10, 10],
                "packets_lost": [0, 1, 0],
                "rtt_ms": [20.0, 80.0, 25.0],
                "retransmissions": [0, 2, 0],
                "resets": [0, 0, 0],
            }
        ).to_csv(run_dir / "sensor_metrics_raw.csv", index=False)

    output_path = tmp_path / "combined.csv"
    combined = build_sensor_experiment_dataset(
        [run_a, run_b],
        output_path=output_path,
    )

    assert output_path.exists()
    assert set(combined["run_id"]) == {"run-a", "run-b"}
    assert "phase_name" in combined.columns
    assert "severity" in combined.columns
    assert json.loads((run_a / "manifest.json").read_text(encoding="utf-8"))["run_id"] == "run-a"
