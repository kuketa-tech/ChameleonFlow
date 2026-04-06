from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.training.generate_controlled_sensor_domains import (
    build_controlled_sensor_domain_frame,
    generate_controlled_sensor_domains,
)


def test_build_controlled_sensor_domain_frame_emits_both_classes() -> None:
    frame = build_controlled_sensor_domain_frame(
        "fiber_lab",
        sessions_per_application=2,
        baseline_rows=5,
        impairment_rows=5,
        recovery_rows=3,
        seed=7,
    )

    assert not frame.empty
    assert set(frame["label"].unique()) == {0, 1}
    assert {"domain", "application", "impairment_type", "severity", "phase_name"} <= set(frame.columns)


def test_generate_controlled_sensor_domains_writes_csvs_and_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "controlled"
    summary_path = tmp_path / "summary.json"

    summary = generate_controlled_sensor_domains(
        output_dir=output_dir,
        domains=["fiber_lab", "home_wifi"],
        applications=["browsing", "voice"],
        sessions_per_application=2,
        baseline_rows=5,
        impairment_rows=5,
        recovery_rows=3,
        seed=11,
        summary_path=summary_path,
    )

    assert summary_path.exists()
    restored = json.loads(summary_path.read_text(encoding="utf-8"))
    assert restored["domains"] == ["fiber_lab", "home_wifi"]

    for domain_name in restored["domains"]:
        domain_path = output_dir / f"{domain_name}.csv"
        assert domain_path.exists()
        frame = pd.read_csv(domain_path)
        assert len(frame) == summary["rows_by_domain"][domain_name]
        assert frame["session_id"].nunique() == summary["sessions_by_domain"][domain_name]
