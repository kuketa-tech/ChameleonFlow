from __future__ import annotations

import pandas as pd

from ml.training.prepare_cicids2017_sensor_metrics import build_cicids2017_sensor_metrics


def test_build_cicids2017_sensor_metrics_converts_flow_rows(tmp_path) -> None:
    input_path = tmp_path / "monday.csv"
    pd.DataFrame(
        {
            "Total Fwd Packets": [10, 20],
            "Total Backward Packets": [1, 5],
            "Total Length of Fwd Packets": [1000, 2000],
            "Total Length of Bwd Packets": [100, 600],
            "Flow Duration": [20_000.0, 40_000.0],
            "Flow Packets/s": [550.0, 625.0],
            "Flow Bytes/s": [8_000.0, 10_000.0],
            "RST Flag Count": [0, 2],
            "Flow IAT Mean": [30_000.0, 180_000.0],
            "Flow IAT Std": [5_000.0, 50_000.0],
            "Packet Length Mean": [80.0, 110.0],
            "Packet Length Std": [10.0, 20.0],
            "Average Packet Size": [75.0, 105.0],
            "Down/Up Ratio": [0.1, 0.4],
            "SYN Flag Count": [1, 3],
            "ACK Flag Count": [10, 20],
            "PSH Flag Count": [0, 2],
            "Active Mean": [1_000.0, 2_000.0],
            "Idle Mean": [500.0, 1_500.0],
            "act_data_pkt_fwd": [9, 14],
            "Label": ["BENIGN", "PortScan"],
        }
    ).to_csv(input_path, index=False)

    prepared = build_cicids2017_sensor_metrics([input_path])

    assert len(prepared) == 2
    assert "flow_packets_per_second" in prepared.columns
    assert "packet_length_std" in prepared.columns
    assert "ack_flag_count" in prepared.columns
    assert prepared["label"].tolist() == [0, 1]
    assert prepared["packets_sent"].tolist() == [11, 25]
    assert prepared["flow_duration_ms"].tolist() == [20.0, 40.0]
