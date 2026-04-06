from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.training.prepare_sensor_metrics import save_sensor_metrics_frame


@dataclass(frozen=True, slots=True)
class ControlledDomainProfile:
    name: str
    base_rtt_ms: float
    rtt_jitter_ms: float
    base_packets_per_second: float
    base_packet_size: float
    base_loss_ratio: float
    direction_ratio: float
    base_iat_cv: float
    active_idle_ratio: float
    noise_scale: float


@dataclass(frozen=True, slots=True)
class ApplicationProfile:
    name: str
    pps_multiplier: float
    packet_size_multiplier: float
    packet_length_cv: float
    direction_ratio_multiplier: float
    psh_ratio: float
    burstiness: float


CONTROLLED_DOMAIN_PROFILES: dict[str, ControlledDomainProfile] = {
    "fiber_lab": ControlledDomainProfile(
        name="fiber_lab",
        base_rtt_ms=14.0,
        rtt_jitter_ms=2.5,
        base_packets_per_second=165.0,
        base_packet_size=1_250.0,
        base_loss_ratio=0.001,
        direction_ratio=1.3,
        base_iat_cv=0.12,
        active_idle_ratio=3.4,
        noise_scale=0.05,
    ),
    "home_wifi": ControlledDomainProfile(
        name="home_wifi",
        base_rtt_ms=28.0,
        rtt_jitter_ms=8.0,
        base_packets_per_second=92.0,
        base_packet_size=980.0,
        base_loss_ratio=0.004,
        direction_ratio=1.9,
        base_iat_cv=0.26,
        active_idle_ratio=2.0,
        noise_scale=0.10,
    ),
    "lte_edge": ControlledDomainProfile(
        name="lte_edge",
        base_rtt_ms=62.0,
        rtt_jitter_ms=14.0,
        base_packets_per_second=58.0,
        base_packet_size=760.0,
        base_loss_ratio=0.006,
        direction_ratio=1.1,
        base_iat_cv=0.34,
        active_idle_ratio=1.5,
        noise_scale=0.12,
    ),
    "public_hotspot": ControlledDomainProfile(
        name="public_hotspot",
        base_rtt_ms=46.0,
        rtt_jitter_ms=18.0,
        base_packets_per_second=74.0,
        base_packet_size=860.0,
        base_loss_ratio=0.008,
        direction_ratio=1.6,
        base_iat_cv=0.38,
        active_idle_ratio=1.3,
        noise_scale=0.15,
    ),
    "satellite_emulated": ControlledDomainProfile(
        name="satellite_emulated",
        base_rtt_ms=185.0,
        rtt_jitter_ms=20.0,
        base_packets_per_second=34.0,
        base_packet_size=690.0,
        base_loss_ratio=0.003,
        direction_ratio=1.0,
        base_iat_cv=0.18,
        active_idle_ratio=1.2,
        noise_scale=0.08,
    ),
}

APPLICATION_PROFILES: dict[str, ApplicationProfile] = {
    "browsing": ApplicationProfile(
        name="browsing",
        pps_multiplier=0.45,
        packet_size_multiplier=0.75,
        packet_length_cv=0.32,
        direction_ratio_multiplier=2.1,
        psh_ratio=0.08,
        burstiness=1.35,
    ),
    "streaming": ApplicationProfile(
        name="streaming",
        pps_multiplier=1.35,
        packet_size_multiplier=1.15,
        packet_length_cv=0.18,
        direction_ratio_multiplier=5.2,
        psh_ratio=0.03,
        burstiness=0.85,
    ),
    "bulk_transfer": ApplicationProfile(
        name="bulk_transfer",
        pps_multiplier=1.65,
        packet_size_multiplier=1.20,
        packet_length_cv=0.12,
        direction_ratio_multiplier=1.0,
        psh_ratio=0.05,
        burstiness=0.95,
    ),
    "voice": ApplicationProfile(
        name="voice",
        pps_multiplier=0.72,
        packet_size_multiplier=0.22,
        packet_length_cv=0.20,
        direction_ratio_multiplier=1.1,
        psh_ratio=0.01,
        burstiness=0.70,
    ),
}

SEVERITY_FACTORS = {
    "mild": 0.55,
    "moderate": 1.0,
    "severe": 1.5,
}

IMPAIRMENT_PRESETS: dict[str, dict[str, float]] = {
    "delay": {
        "rtt_add_ms": 85.0,
        "jitter_multiplier": 1.9,
        "loss_add_ratio": 0.004,
        "pps_multiplier": 0.92,
        "byte_rate_multiplier": 0.93,
        "retransmission_spike": 0.4,
        "reset_probability": 0.002,
        "iat_cv_add": 0.10,
        "active_idle_multiplier": 0.82,
    },
    "jitter": {
        "rtt_add_ms": 28.0,
        "jitter_multiplier": 4.8,
        "loss_add_ratio": 0.006,
        "pps_multiplier": 0.88,
        "byte_rate_multiplier": 0.90,
        "retransmission_spike": 0.8,
        "reset_probability": 0.004,
        "iat_cv_add": 0.28,
        "active_idle_multiplier": 0.76,
    },
    "loss": {
        "rtt_add_ms": 36.0,
        "jitter_multiplier": 2.8,
        "loss_add_ratio": 0.08,
        "pps_multiplier": 0.82,
        "byte_rate_multiplier": 0.80,
        "retransmission_spike": 3.4,
        "reset_probability": 0.03,
        "iat_cv_add": 0.16,
        "active_idle_multiplier": 0.72,
    },
    "bandwidth": {
        "rtt_add_ms": 48.0,
        "jitter_multiplier": 2.1,
        "loss_add_ratio": 0.010,
        "pps_multiplier": 0.54,
        "byte_rate_multiplier": 0.44,
        "retransmission_spike": 1.2,
        "reset_probability": 0.010,
        "iat_cv_add": 0.12,
        "active_idle_multiplier": 0.64,
    },
    "mixed": {
        "rtt_add_ms": 96.0,
        "jitter_multiplier": 5.8,
        "loss_add_ratio": 0.060,
        "pps_multiplier": 0.60,
        "byte_rate_multiplier": 0.52,
        "retransmission_spike": 4.2,
        "reset_probability": 0.040,
        "iat_cv_add": 0.34,
        "active_idle_multiplier": 0.58,
    },
}

IMPAIRMENT_SEQUENCE = (
    ("delay", "mild"),
    ("jitter", "moderate"),
    ("loss", "moderate"),
    ("bandwidth", "severe"),
    ("mixed", "moderate"),
    ("delay", "severe"),
    ("loss", "mild"),
    ("mixed", "severe"),
)

RAW_SENSOR_OUTPUT_COLUMNS = [
    "session_id",
    "timestamp",
    "packets_sent",
    "packets_lost",
    "rtt_ms",
    "retransmissions",
    "resets",
    "label",
    "flow_duration_ms",
    "flow_packets_per_second",
    "flow_bytes_per_second",
    "average_packet_size",
    "packet_length_mean",
    "packet_length_std",
    "fwd_packet_count",
    "bwd_packet_count",
    "fwd_byte_count",
    "bwd_byte_count",
    "down_up_ratio",
    "flow_iat_mean_us",
    "flow_iat_std_us",
    "active_mean_us",
    "idle_mean_us",
    "syn_flag_count",
    "ack_flag_count",
    "psh_flag_count",
    "domain",
    "application",
    "impairment_type",
    "severity",
    "phase_name",
]


def controlled_domain_names() -> list[str]:
    return list(CONTROLLED_DOMAIN_PROFILES)


def _baseline_state(
    domain: ControlledDomainProfile,
    application: ApplicationProfile,
    *,
    session_bias: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    packets_per_second = max(
        6.0,
        domain.base_packets_per_second
        * application.pps_multiplier
        * (1.0 + session_bias * 0.10)
        * (1.0 + rng.normal(0.0, domain.noise_scale * 0.30)),
    )
    average_packet_size = max(
        64.0,
        domain.base_packet_size
        * application.packet_size_multiplier
        * (1.0 + rng.normal(0.0, domain.noise_scale * 0.25)),
    )
    return {
        "packets_per_second": packets_per_second,
        "rtt_ms": max(
            1.0,
            domain.base_rtt_ms
            * (1.0 + session_bias * 0.04)
            * (1.0 + rng.normal(0.0, domain.noise_scale * 0.18)),
        ),
        "rtt_jitter_ms": max(
            0.5,
            domain.rtt_jitter_ms
            * (1.0 + session_bias * 0.08)
            * (1.0 + rng.normal(0.0, domain.noise_scale * 0.25)),
        ),
        "loss_ratio": float(
            np.clip(
                domain.base_loss_ratio
                * (1.0 + session_bias * 0.15)
                * (1.0 + rng.normal(0.0, domain.noise_scale * 0.35)),
                0.0,
                0.25,
            )
        ),
        "byte_rate": packets_per_second * average_packet_size,
        "average_packet_size": average_packet_size,
        "packet_length_cv": max(
            0.04,
            application.packet_length_cv * (1.0 + rng.normal(0.0, domain.noise_scale * 0.20)),
        ),
        "direction_ratio": max(
            0.25,
            domain.direction_ratio
            * application.direction_ratio_multiplier
            * (1.0 + rng.normal(0.0, domain.noise_scale * 0.18)),
        ),
        "iat_cv": max(
            0.02,
            domain.base_iat_cv
            * application.burstiness
            * (1.0 + rng.normal(0.0, domain.noise_scale * 0.30)),
        ),
        "active_idle_ratio": max(
            0.10,
            domain.active_idle_ratio
            * application.burstiness
            * (1.0 + rng.normal(0.0, domain.noise_scale * 0.30)),
        ),
        "ack_ratio": float(np.clip(0.72 + rng.normal(0.0, 0.05), 0.35, 1.1)),
        "psh_ratio": float(
            np.clip(
                application.psh_ratio * (1.0 + rng.normal(0.0, domain.noise_scale * 0.60)),
                0.0,
                0.35,
            )
        ),
        "flow_duration_ms": max(
            80.0,
            (900.0 + 250.0 * application.burstiness) * (1.0 + rng.normal(0.0, 0.08)),
        ),
    }


def _apply_impairment(
    baseline: dict[str, float],
    *,
    impairment_type: str,
    severity: str,
    strength: float,
    domain_noise_scale: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    preset = IMPAIRMENT_PRESETS[impairment_type]
    factor = SEVERITY_FACTORS[severity] * strength
    impaired = dict(baseline)
    impaired["rtt_ms"] = baseline["rtt_ms"] + (preset["rtt_add_ms"] * factor)
    impaired["rtt_jitter_ms"] = baseline["rtt_jitter_ms"] * (
        1.0 + (preset["jitter_multiplier"] - 1.0) * factor
    )
    impaired["loss_ratio"] = float(
        np.clip(baseline["loss_ratio"] + (preset["loss_add_ratio"] * factor), 0.0, 0.45)
    )
    impaired["packets_per_second"] = max(
        2.0,
        baseline["packets_per_second"]
        * (1.0 - (1.0 - preset["pps_multiplier"]) * factor)
        * (1.0 + rng.normal(0.0, domain_noise_scale * 0.20)),
    )
    impaired["byte_rate"] = max(
        512.0,
        baseline["byte_rate"]
        * (1.0 - (1.0 - preset["byte_rate_multiplier"]) * factor)
        * (1.0 + rng.normal(0.0, domain_noise_scale * 0.18)),
    )
    impaired["iat_cv"] = max(0.03, baseline["iat_cv"] + (preset["iat_cv_add"] * factor))
    impaired["active_idle_ratio"] = max(
        0.05,
        baseline["active_idle_ratio"]
        * (1.0 - (1.0 - preset["active_idle_multiplier"]) * factor),
    )
    impaired["retransmission_spike"] = preset["retransmission_spike"] * factor
    impaired["reset_probability"] = preset["reset_probability"] * factor
    return impaired


def _build_controlled_row(
    *,
    session_id: str,
    timestamp: pd.Timestamp,
    baseline: dict[str, float],
    impairment_type: str,
    severity: str,
    phase_name: str,
    phase_progress: float,
    domain_name: str,
    application_name: str,
    rng: np.random.Generator,
    domain_noise_scale: float,
) -> dict[str, Any]:
    if phase_name == "baseline":
        state = dict(baseline)
        label = 0
        impairment_strength = 0.0
    elif phase_name == "impairment":
        state = _apply_impairment(
            baseline,
            impairment_type=impairment_type,
            severity=severity,
            strength=1.0,
            domain_noise_scale=domain_noise_scale,
            rng=rng,
        )
        label = 1
        impairment_strength = 1.0
    else:
        impairment_strength = max(0.0, 1.0 - phase_progress * 1.10)
        state = _apply_impairment(
            baseline,
            impairment_type=impairment_type,
            severity=severity,
            strength=impairment_strength,
            domain_noise_scale=domain_noise_scale,
            rng=rng,
        )
        label = 0

    packets_per_second = max(
        2.0,
        state["packets_per_second"] * (1.0 + rng.normal(0.0, domain_noise_scale * 0.18)),
    )
    packets_sent = max(1, int(rng.poisson(packets_per_second)))
    packets_lost = int(
        rng.binomial(
            n=max(packets_sent, 1),
            p=float(np.clip(state["loss_ratio"], 0.0, 0.90)),
        )
    )
    total_packets = max(1.0, float(packets_sent + packets_lost))
    rtt_ms = max(
        1.0,
        float(rng.normal(state["rtt_ms"], max(state["rtt_jitter_ms"], 0.75))),
    )
    retransmissions = int(
        rng.poisson(
            max(
                0.0,
                packets_sent * state["loss_ratio"] * 0.18
                + state.get("retransmission_spike", 0.0)
                + (impairment_strength * 0.3),
            )
        )
    )
    resets = int(rng.random() < state.get("reset_probability", 0.0))
    average_packet_size = max(
        64.0,
        state["average_packet_size"] * (1.0 + rng.normal(0.0, domain_noise_scale * 0.10)),
    )
    flow_packets_per_second = max(
        1.0,
        packets_per_second * (1.0 + rng.normal(0.0, domain_noise_scale * 0.08)),
    )
    flow_bytes_per_second = max(
        256.0,
        state["byte_rate"] * (1.0 + rng.normal(0.0, domain_noise_scale * 0.12)),
    )
    packet_length_mean = average_packet_size * (1.0 + rng.normal(0.0, domain_noise_scale * 0.04))
    packet_length_std = max(
        2.0,
        packet_length_mean
        * state["packet_length_cv"]
        * (1.0 + rng.normal(0.0, domain_noise_scale * 0.12)),
    )
    direction_ratio = max(
        0.20,
        state["direction_ratio"] * (1.0 + rng.normal(0.0, domain_noise_scale * 0.10)),
    )
    bwd_packet_count = max(1.0, total_packets / (1.0 + direction_ratio))
    fwd_packet_count = max(1.0, total_packets - bwd_packet_count)
    bwd_byte_count = max(64.0, bwd_packet_count * average_packet_size * 0.85)
    fwd_byte_count = max(64.0, fwd_packet_count * average_packet_size * 1.05)
    flow_iat_mean_us = max(10.0, 1_000_000.0 / flow_packets_per_second)
    flow_iat_std_us = max(
        1.0,
        flow_iat_mean_us
        * state["iat_cv"]
        * (1.0 + rng.normal(0.0, domain_noise_scale * 0.14)),
    )
    idle_mean_us = max(10.0, flow_iat_mean_us * (1.0 + rng.normal(0.0, 0.08)))
    active_mean_us = max(
        10.0,
        idle_mean_us
        * state["active_idle_ratio"]
        * (1.0 + rng.normal(0.0, domain_noise_scale * 0.10)),
    )
    syn_flag_count = max(
        0.0,
        rng.poisson(0.12 + (resets * 1.7) + (impairment_strength * 0.4)),
    )
    ack_flag_count = max(1.0, total_packets * state["ack_ratio"])
    psh_flag_count = max(0.0, total_packets * state["psh_ratio"])

    return {
        "session_id": session_id,
        "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
        "packets_sent": packets_sent,
        "packets_lost": packets_lost,
        "rtt_ms": rtt_ms,
        "retransmissions": retransmissions,
        "resets": resets,
        "label": label,
        "flow_duration_ms": state["flow_duration_ms"] * (1.0 + rng.normal(0.0, 0.04)),
        "flow_packets_per_second": flow_packets_per_second,
        "flow_bytes_per_second": flow_bytes_per_second,
        "average_packet_size": average_packet_size,
        "packet_length_mean": packet_length_mean,
        "packet_length_std": packet_length_std,
        "fwd_packet_count": fwd_packet_count,
        "bwd_packet_count": bwd_packet_count,
        "fwd_byte_count": fwd_byte_count,
        "bwd_byte_count": bwd_byte_count,
        "down_up_ratio": fwd_packet_count / max(bwd_packet_count, 1.0),
        "flow_iat_mean_us": flow_iat_mean_us,
        "flow_iat_std_us": flow_iat_std_us,
        "active_mean_us": active_mean_us,
        "idle_mean_us": idle_mean_us,
        "syn_flag_count": syn_flag_count,
        "ack_flag_count": ack_flag_count,
        "psh_flag_count": psh_flag_count,
        "domain": domain_name,
        "application": application_name,
        "impairment_type": impairment_type,
        "severity": severity,
        "phase_name": phase_name,
    }


def build_controlled_sensor_domain_frame(
    domain_name: str,
    *,
    applications: Sequence[str] | None = None,
    sessions_per_application: int = 24,
    baseline_rows: int = 25,
    impairment_rows: int = 25,
    recovery_rows: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    try:
        domain = CONTROLLED_DOMAIN_PROFILES[domain_name]
    except KeyError as exc:
        msg = f"Unknown controlled domain: {domain_name}"
        raise ValueError(msg) from exc

    selected_applications = list(applications or APPLICATION_PROFILES)
    rows: list[dict[str, Any]] = []
    domain_index = controlled_domain_names().index(domain_name)
    domain_rng = np.random.default_rng(seed + (domain_index * 10_000))
    domain_start = pd.Timestamp("2026-04-06T10:00:00Z") + pd.Timedelta(days=domain_index)
    session_stride_seconds = baseline_rows + impairment_rows + recovery_rows + 30

    for app_index, application_name in enumerate(selected_applications):
        application = APPLICATION_PROFILES[application_name]
        for session_index in range(sessions_per_application):
            session_id = f"{domain_name}-{application_name}-{session_index:04d}"
            impairment_type, severity = IMPAIRMENT_SEQUENCE[
                (session_index + app_index) % len(IMPAIRMENT_SEQUENCE)
            ]
            baseline = _baseline_state(
                domain,
                application,
                session_bias=float(domain_rng.normal(0.0, 1.0)),
                rng=domain_rng,
            )
            session_start = domain_start + pd.Timedelta(
                seconds=((app_index * sessions_per_application) + session_index) * session_stride_seconds
            )

            phase_specs = (
                ("baseline", baseline_rows),
                ("impairment", impairment_rows),
                ("recovery", recovery_rows),
            )
            second_offset = 0
            for phase_name, phase_rows in phase_specs:
                for row_index in range(phase_rows):
                    phase_progress = row_index / max(phase_rows - 1, 1)
                    timestamp = session_start + pd.Timedelta(seconds=second_offset)
                    rows.append(
                        _build_controlled_row(
                            session_id=session_id,
                            timestamp=timestamp,
                            baseline=baseline,
                            impairment_type=impairment_type,
                            severity=severity,
                            phase_name=phase_name,
                            phase_progress=phase_progress,
                            domain_name=domain_name,
                            application_name=application_name,
                            rng=domain_rng,
                            domain_noise_scale=domain.noise_scale,
                        )
                    )
                    second_offset += 1

    return pd.DataFrame(rows, columns=RAW_SENSOR_OUTPUT_COLUMNS)


def generate_controlled_sensor_domains(
    *,
    output_dir: Path,
    domains: Sequence[str] | None = None,
    applications: Sequence[str] | None = None,
    sessions_per_application: int = 24,
    baseline_rows: int = 25,
    impairment_rows: int = 25,
    recovery_rows: int = 15,
    seed: int = 42,
    summary_path: Path | None = None,
) -> dict[str, Any]:
    selected_domains = list(domains or controlled_domain_names())
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_paths: list[str] = []
    rows_by_domain: dict[str, int] = {}
    sessions_by_domain: dict[str, int] = {}

    for domain_name in selected_domains:
        frame = build_controlled_sensor_domain_frame(
            domain_name,
            applications=applications,
            sessions_per_application=sessions_per_application,
            baseline_rows=baseline_rows,
            impairment_rows=impairment_rows,
            recovery_rows=recovery_rows,
            seed=seed,
        )
        output_path = output_dir / f"{domain_name}.csv"
        save_sensor_metrics_frame(frame, output_path)
        generated_paths.append(str(output_path))
        rows_by_domain[domain_name] = int(len(frame))
        sessions_by_domain[domain_name] = int(frame["session_id"].nunique())

    summary = {
        "generator": "controlled_sensor_domains_v1",
        "domains": selected_domains,
        "applications": list(applications or APPLICATION_PROFILES),
        "sessions_per_application": sessions_per_application,
        "baseline_rows": baseline_rows,
        "impairment_rows": impairment_rows,
        "recovery_rows": recovery_rows,
        "rows_by_domain": rows_by_domain,
        "sessions_by_domain": sessions_by_domain,
        "generated_paths": generated_paths,
        "seed": seed,
    }
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
