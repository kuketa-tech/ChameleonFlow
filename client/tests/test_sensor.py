from __future__ import annotations

from client.app.sensor.service import ChannelMetricsWindow, ThresholdSensor


def test_threshold_sensor_marks_healthy_window_as_not_degraded() -> None:
    sensor = ThresholdSensor(threshold=0.7)

    result = sensor.evaluate(
        ChannelMetricsWindow(
            packet_loss_ratio=0.05,
            rtt_variation_ratio=0.10,
            retransmission_ratio=0.10,
            reset_ratio=0.0,
        )
    )

    assert result.degraded is False
    assert result.probability < 0.7


def test_threshold_sensor_marks_bad_window_as_degraded() -> None:
    sensor = ThresholdSensor(threshold=0.7)

    result = sensor.evaluate(
        ChannelMetricsWindow(
            packet_loss_ratio=0.80,
            rtt_variation_ratio=0.90,
            retransmission_ratio=0.70,
            reset_ratio=0.60,
        )
    )

    assert result.degraded is True
    assert result.probability >= 0.7
