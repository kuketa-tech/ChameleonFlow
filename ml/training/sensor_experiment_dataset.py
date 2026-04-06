from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ml.training.prepare_sensor_metrics import build_sensor_metrics_frame, save_sensor_metrics_frame


class SensorExperimentFiles(BaseModel):
    model_config = ConfigDict(frozen=True)

    raw_metrics: str = "sensor_metrics_raw.csv"
    pcap: str | None = "capture.pcap"
    probes: str | None = "probes.csv"
    transport_stats: str | None = "transport_stats.csv"


class SensorExperimentPhase(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    start_offset_seconds: float = Field(ge=0.0)
    duration_seconds: float = Field(gt=0.0)
    label: int = Field(ge=0)
    impairment_type: str
    severity: Literal["none", "mild", "moderate", "severe"] = "none"
    notes: str | None = None
    netem: dict[str, Any] | None = None

    @property
    def end_offset_seconds(self) -> float:
        return self.start_offset_seconds + self.duration_seconds


class SensorExperimentManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    run_id: str
    created_at: datetime
    domain: str
    application: str
    scenario: str
    operator: str | None = None
    host: str | None = None
    interface: str | None = None
    window_seconds: int = 5
    files: SensorExperimentFiles = Field(default_factory=SensorExperimentFiles)
    phases: list[SensorExperimentPhase]

    def manifest_path(self, run_dir: Path) -> Path:
        return run_dir / "manifest.json"


def create_sensor_experiment_manifest(
    *,
    run_id: str,
    domain: str,
    application: str,
    scenario: str,
    operator: str | None = None,
    host: str | None = None,
    interface: str | None = None,
    window_seconds: int = 5,
    baseline_seconds: float = 60.0,
    impairment_seconds: float = 120.0,
    recovery_seconds: float = 60.0,
) -> SensorExperimentManifest:
    phases = [
        SensorExperimentPhase(
            name="baseline",
            start_offset_seconds=0.0,
            duration_seconds=baseline_seconds,
            label=0,
            impairment_type="baseline",
            severity="none",
            notes="Healthy warmup/baseline segment.",
        ),
        SensorExperimentPhase(
            name="impairment",
            start_offset_seconds=baseline_seconds,
            duration_seconds=impairment_seconds,
            label=1,
            impairment_type="netem",
            severity="moderate",
            notes="Replace with the exact applied impairment parameters before final training.",
            netem={
                "delay_ms": 120,
                "jitter_ms": 20,
                "loss_percent": 1.0,
            },
        ),
        SensorExperimentPhase(
            name="recovery",
            start_offset_seconds=baseline_seconds + impairment_seconds,
            duration_seconds=recovery_seconds,
            label=0,
            impairment_type="baseline",
            severity="none",
            notes="Recovery/cooldown segment after impairment removal.",
        ),
    ]
    return SensorExperimentManifest(
        run_id=run_id,
        created_at=datetime.now(tz=UTC),
        domain=domain,
        application=application,
        scenario=scenario,
        operator=operator,
        host=host,
        interface=interface,
        window_seconds=window_seconds,
        phases=phases,
    )


def initialize_sensor_experiment_run(
    *,
    run_dir: Path,
    run_id: str,
    domain: str,
    application: str,
    scenario: str,
    operator: str | None = None,
    host: str | None = None,
    interface: str | None = None,
    window_seconds: int = 5,
    baseline_seconds: float = 60.0,
    impairment_seconds: float = 120.0,
    recovery_seconds: float = 60.0,
) -> SensorExperimentManifest:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = create_sensor_experiment_manifest(
        run_id=run_id,
        domain=domain,
        application=application,
        scenario=scenario,
        operator=operator,
        host=host,
        interface=interface,
        window_seconds=window_seconds,
        baseline_seconds=baseline_seconds,
        impairment_seconds=impairment_seconds,
        recovery_seconds=recovery_seconds,
    )
    save_sensor_experiment_manifest(manifest, run_dir / "manifest.json")
    raw_metrics_path = run_dir / manifest.files.raw_metrics
    if not raw_metrics_path.exists():
        raw_metrics_path.write_text(
            (
                "session_id,timestamp,packets_sent,packets_lost,rtt_ms,"
                "retransmissions,resets\n"
            ),
            encoding="utf-8",
        )
    return manifest


def save_sensor_experiment_manifest(manifest: SensorExperimentManifest, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_sensor_experiment_manifest(path: Path) -> SensorExperimentManifest:
    return SensorExperimentManifest.model_validate_json(path.read_text(encoding="utf-8"))


def label_sensor_experiment_frame(
    *,
    raw_frame: pd.DataFrame,
    manifest: SensorExperimentManifest,
    strict_coverage: bool = True,
) -> pd.DataFrame:
    prepared = build_sensor_metrics_frame(
        raw_frame,
        require_label=False,
    ).copy()
    if prepared.empty:
        return prepared

    capture_start = prepared["timestamp"].min()
    prepared["run_id"] = manifest.run_id
    prepared["domain"] = manifest.domain
    prepared["application"] = manifest.application
    prepared["scenario"] = manifest.scenario
    prepared["phase_name"] = "unlabeled"
    prepared["impairment_type"] = "unlabeled"
    prepared["severity"] = "none"
    prepared["capture_offset_seconds"] = (
        prepared["timestamp"] - capture_start
    ).dt.total_seconds()

    covered = pd.Series(False, index=prepared.index)
    for phase in manifest.phases:
        phase_mask = prepared["capture_offset_seconds"].ge(phase.start_offset_seconds) & prepared[
            "capture_offset_seconds"
        ].lt(phase.end_offset_seconds)
        if not phase_mask.any():
            continue
        prepared.loc[phase_mask, "label"] = phase.label
        prepared.loc[phase_mask, "phase_name"] = phase.name
        prepared.loc[phase_mask, "impairment_type"] = phase.impairment_type
        prepared.loc[phase_mask, "severity"] = phase.severity
        covered |= phase_mask

    if strict_coverage and not bool(covered.all()):
        uncovered_rows = prepared.loc[~covered, ["timestamp", "capture_offset_seconds"]].head(5)
        msg = (
            "Manifest phases do not cover every sensor row. "
            f"Example uncovered rows: {uncovered_rows.to_dict(orient='records')}"
        )
        raise ValueError(msg)

    return prepared


def build_sensor_experiment_dataset(
    run_dirs: list[Path],
    *,
    output_path: Path,
    strict_coverage: bool = True,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        manifest = load_sensor_experiment_manifest(run_dir / "manifest.json")
        raw_metrics_path = run_dir / manifest.files.raw_metrics
        raw_frame = pd.read_csv(raw_metrics_path)
        frames.append(
            label_sensor_experiment_frame(
                raw_frame=raw_frame,
                manifest=manifest,
                strict_coverage=strict_coverage,
            )
        )

    if not frames:
        msg = "At least one run directory is required."
        raise ValueError(msg)

    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["run_id", "session_id", "timestamp"]
    ).reset_index(drop=True)
    save_sensor_metrics_frame(combined, output_path)
    return combined
