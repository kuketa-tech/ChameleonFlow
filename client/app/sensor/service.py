from __future__ import annotations

from dataclasses import dataclass

from shared.contracts import SensorResult


@dataclass(slots=True)
class ChannelMetricsWindow:
    packet_loss_ratio: float
    rtt_variation_ratio: float
    retransmission_ratio: float
    reset_ratio: float


class ThresholdSensor:
    def __init__(self, threshold: float = 0.7) -> None:
        self._threshold = threshold

    def evaluate(self, metrics: ChannelMetricsWindow) -> SensorResult:
        score = (
            metrics.packet_loss_ratio
            + metrics.rtt_variation_ratio
            + metrics.retransmission_ratio
            + metrics.reset_ratio
        ) / 4
        probability = min(max(score, 0.0), 1.0)
        return SensorResult(
            probability=probability,
            degraded=probability >= self._threshold,
        )
