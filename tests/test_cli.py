from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_list_transports_outputs_supported_transport_names() -> None:
    result = subprocess.run(
        [sys.executable, "main.py", "list-transports"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout.splitlines() == ["doh", "webrtc", "quic"]


def test_prepare_browser_iat_command_writes_rows(tmp_path) -> None:
    input_path = tmp_path / "packets.csv"
    output_path = tmp_path / "browser_iat.csv"
    pd.DataFrame(
        {
            "trace_id": ["a", "a", "a"],
            "packet_index": [0, 1, 2],
            "timestamp": [0.0, 0.1, 0.3],
        }
    ).to_csv(input_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            "prepare-browser-iat",
            str(input_path),
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert output_path.exists()
    assert "prepared 2 rows" in result.stdout


def test_generate_sensor_sample_command_writes_rows(tmp_path) -> None:
    output_path = tmp_path / "sensor_metrics.csv"

    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            "generate-sensor-sample",
            str(output_path),
            "--sessions",
            "10",
            "--rows-per-session",
            "4",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert output_path.exists()
    assert "generated 40 sensor rows" in result.stdout


def test_prepare_browser_iat_from_pcap_help_lists_max_iat_option() -> None:
    result = subprocess.run(
        [sys.executable, "main.py", "prepare-browser-iat-from-pcap", "--help"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--max-iat-ms" in result.stdout


def test_prepare_cicids2017_sensor_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "main.py", "prepare-cicids2017-sensor", "--help"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--input" in result.stdout
    assert "--output" in result.stdout


def test_prepare_iscxvpn2016_sensor_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "main.py", "prepare-iscxvpn2016-sensor", "--help"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--input" in result.stdout
    assert "--positive-pattern" in result.stdout


def test_train_morpher_help_lists_device_option() -> None:
    result = subprocess.run(
        [sys.executable, "main.py", "train-morpher", "--help"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--device" in result.stdout


def test_train_sensor_help_lists_threshold_option() -> None:
    result = subprocess.run(
        [sys.executable, "main.py", "train-sensor", "--help"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--threshold" in result.stdout


def test_evaluate_sensor_transfer_help_lists_eval_input_option() -> None:
    result = subprocess.run(
        [sys.executable, "main.py", "evaluate-sensor-transfer", "--help"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--algorithm" in result.stdout
    assert "--max-eval-rows" in result.stdout
