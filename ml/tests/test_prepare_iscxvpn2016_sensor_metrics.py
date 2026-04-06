from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.training.prepare_iscxvpn2016_sensor_metrics import build_iscxvpn2016_sensor_metrics


def test_build_iscxvpn2016_sensor_metrics_converts_csv_rows(tmp_path: Path) -> None:
    input_path = tmp_path / "vpn.csv"
    pd.DataFrame(
        {
            "Flow ID": ["flow-1", "flow-2"],
            "Total Fwd Packets": [10, 30],
            "Total Backward Packets": [2, 6],
            "Total Length of Fwd Packets": [1000, 2200],
            "Total Length of Bwd Packets": [200, 500],
            "Flow Duration": [20_000.0, 80_000.0],
            "Flow Packets/s": [600.0, 450.0],
            "Flow Bytes/s": [9_000.0, 11_000.0],
            "Flow IAT Mean": [20_000.0, 50_000.0],
            "Flow IAT Std": [5_000.0, 12_000.0],
            "Packet Length Mean": [80.0, 90.0],
            "Packet Length Std": [10.0, 14.0],
            "Average Packet Size": [75.0, 88.0],
            "RST Flag Count": [0, 1],
            "SYN Flag Count": [1, 2],
            "ACK Flag Count": [10, 30],
            "PSH Flag Count": [0, 2],
            "Down/Up Ratio": [0.2, 0.3],
            "act_data_pkt_fwd": [9, 24],
            "Label": ["NonVPN-Web", "VPN-Web"],
        }
    ).to_csv(input_path, index=False)

    prepared = build_iscxvpn2016_sensor_metrics([input_path])

    assert len(prepared) == 2
    assert prepared["session_id"].tolist() == ["flow-1", "flow-2"]
    assert prepared["label"].tolist() == [0, 1]
    assert prepared["packets_sent"].tolist() == [12, 36]
    assert prepared["retransmissions"].tolist() == [1, 6]
    assert "flow_bytes_per_second" in prepared.columns


def test_build_iscxvpn2016_sensor_metrics_supports_arff(tmp_path: Path) -> None:
    input_path = tmp_path / "vpn.arff"
    input_path.write_text(
        "\n".join(
            [
                "@RELATION vpn",
                "@ATTRIBUTE flow_id STRING",
                "@ATTRIBUTE total_fwd_packets NUMERIC",
                "@ATTRIBUTE total_bwd_packets NUMERIC",
                "@ATTRIBUTE flow_iat_mean NUMERIC",
                "@ATTRIBUTE flow_iat_std NUMERIC",
                "@ATTRIBUTE rst_flag_count NUMERIC",
                "@ATTRIBUTE class {NonVPN,VPN}",
                "@DATA",
                "'alpha',10,2,20000,4000,0,NonVPN",
                "'beta',20,4,40000,8000,1,VPN",
            ]
        ),
        encoding="utf-8",
    )

    prepared = build_iscxvpn2016_sensor_metrics([input_path])

    assert len(prepared) == 2
    assert prepared["session_id"].tolist() == ["alpha", "beta"]
    assert prepared["label"].tolist() == [0, 1]
    assert prepared["rtt_ms"].tolist() == [20.0, 40.0]
