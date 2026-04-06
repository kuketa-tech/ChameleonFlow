from __future__ import annotations

import argparse
import csv
import re
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ml.training.sensor_experiment_dataset import (
    SensorExperimentManifest,
    SensorExperimentPhase,
    load_sensor_experiment_manifest,
)

PING_REPLY_PATTERN = re.compile(
    r"^(?:\[(?P<epoch>[0-9]+\.[0-9]+)\]\s+)?"
    r".*icmp_seq=(?P<seq>\d+).*time=(?P<rtt_ms>[0-9.]+)\s*ms",
)
PING_TIMEOUT_PATTERN = re.compile(
    r"^(?:\[(?P<epoch>[0-9]+\.[0-9]+)\]\s+)?"
    r"no answer yet for icmp_seq=(?P<seq>\d+)",
)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a minimal ping-based sensor experiment from a manifest.",
    )
    parser.add_argument("--run-dir", required=True, type=Path, help="Experiment run directory")
    parser.add_argument("--target", required=True, help="Ping target host or IP")
    parser.add_argument("--interface", required=True, help="Network interface for tc netem")
    parser.add_argument(
        "--ping-interval-seconds",
        type=float,
        default=0.2,
        help="Interval between ping probes",
    )
    parser.add_argument(
        "--sudo",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Prefix tc commands with sudo",
    )
    return parser


def parse_ping_output_line(line: str) -> dict[str, Any] | None:
    reply_match = PING_REPLY_PATTERN.search(line.strip())
    if reply_match is not None:
        epoch = reply_match.group("epoch")
        return {
            "event_type": "reply",
            "epoch": float(epoch) if epoch is not None else None,
            "icmp_seq": int(reply_match.group("seq")),
            "packets_sent": 1,
            "packets_lost": 0,
            "rtt_ms": float(reply_match.group("rtt_ms")),
            "retransmissions": 0,
            "resets": 0,
        }

    timeout_match = PING_TIMEOUT_PATTERN.search(line.strip())
    if timeout_match is not None:
        epoch = timeout_match.group("epoch")
        return {
            "event_type": "timeout",
            "epoch": float(epoch) if epoch is not None else None,
            "icmp_seq": int(timeout_match.group("seq")),
            "packets_sent": 1,
            "packets_lost": 1,
            "rtt_ms": 0.0,
            "retransmissions": 0,
            "resets": 0,
        }

    return None


def build_netem_tokens(phase: SensorExperimentPhase) -> list[str]:
    config = phase.netem or {}
    if phase.impairment_type == "baseline" or not config:
        return []

    tokens: list[str] = []
    delay_ms = config.get("delay_ms")
    jitter_ms = config.get("jitter_ms")
    if delay_ms is not None:
        tokens.extend(["delay", f"{float(delay_ms):g}ms"])
        if jitter_ms is not None:
            tokens.append(f"{float(jitter_ms):g}ms")

    loss_percent = config.get("loss_percent")
    if loss_percent is not None:
        tokens.extend(["loss", f"{float(loss_percent):g}%"])

    duplicate_percent = config.get("duplicate_percent")
    if duplicate_percent is not None:
        tokens.extend(["duplicate", f"{float(duplicate_percent):g}%"])

    reorder_percent = config.get("reorder_percent")
    if reorder_percent is not None:
        tokens.extend(["reorder", f"{float(reorder_percent):g}%"])

    rate_kbit = config.get("rate_kbit")
    if rate_kbit is not None:
        tokens.extend(["rate", f"{float(rate_kbit):g}kbit"])

    return tokens


def _run_tc_command(arguments: list[str], *, use_sudo: bool) -> None:
    command = ["sudo", *arguments] if use_sudo else arguments
    subprocess.run(command, check=True)


def clear_tc_qdisc(interface: str, *, use_sudo: bool) -> None:
    command = ["tc", "qdisc", "del", "dev", interface, "root"]
    process = subprocess.run(command if not use_sudo else ["sudo", *command], check=False)
    if process.returncode not in {0, 2}:
        process.check_returncode()


def apply_tc_phase(interface: str, phase: SensorExperimentPhase, *, use_sudo: bool) -> None:
    tokens = build_netem_tokens(phase)
    if not tokens:
        clear_tc_qdisc(interface, use_sudo=use_sudo)
        return

    command = ["tc", "qdisc", "replace", "dev", interface, "root", "netem", *tokens]
    _run_tc_command(command, use_sudo=use_sudo)


def _format_event_timestamp(epoch: float | None) -> str:
    if epoch is None:
        return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")

    return datetime.fromtimestamp(epoch, tz=UTC).isoformat().replace("+00:00", "Z")


def run_sensor_ping_experiment(
    *,
    run_dir: Path,
    target: str,
    interface: str,
    ping_interval_seconds: float = 0.2,
    use_sudo: bool = False,
) -> Path:
    manifest = load_sensor_experiment_manifest(run_dir / "manifest.json")
    raw_metrics_path = run_dir / manifest.files.raw_metrics
    raw_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ping",
        "-D",
        "-O",
        "-n",
        "-i",
        str(ping_interval_seconds),
        target,
    ]

    with raw_metrics_path.open("w", encoding="utf-8", newline="") as raw_file:
        writer = csv.DictWriter(
            raw_file,
            fieldnames=[
                "session_id",
                "timestamp",
                "packets_sent",
                "packets_lost",
                "rtt_ms",
                "retransmissions",
                "resets",
            ],
        )
        writer.writeheader()

        ping_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def consume_ping_output() -> None:
            assert ping_process.stdout is not None
            for line in ping_process.stdout:
                event = parse_ping_output_line(line)
                if event is None:
                    continue
                writer.writerow(
                    {
                        "session_id": manifest.run_id,
                        "timestamp": _format_event_timestamp(event["epoch"]),
                        "packets_sent": event["packets_sent"],
                        "packets_lost": event["packets_lost"],
                        "rtt_ms": event["rtt_ms"],
                        "retransmissions": event["retransmissions"],
                        "resets": event["resets"],
                    }
                )
                raw_file.flush()

        reader_thread = threading.Thread(target=consume_ping_output, daemon=True)
        reader_thread.start()

        try:
            for phase in manifest.phases:
                apply_tc_phase(interface, phase, use_sudo=use_sudo)
                time.sleep(phase.duration_seconds)
        finally:
            ping_process.terminate()
            try:
                ping_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ping_process.kill()
                ping_process.wait(timeout=5)
            reader_thread.join(timeout=5)
            clear_tc_qdisc(interface, use_sudo=use_sudo)

    return raw_metrics_path


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()
    raw_metrics_path = run_sensor_ping_experiment(
        run_dir=args.run_dir,
        target=args.target,
        interface=args.interface,
        ping_interval_seconds=args.ping_interval_seconds,
        use_sudo=args.sudo,
    )
    print(raw_metrics_path)


if __name__ == "__main__":
    main()
