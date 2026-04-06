from __future__ import annotations

from shared.contracts import BanditContext, BanditDecision, TransportKind


class InMemoryBanditAgent:
    def __init__(self, model_version: str = "stub-bandit-v1") -> None:
        self._model_version = model_version
        self._scores: dict[TransportKind, float] = {
            TransportKind.QUIC: 0.6,
            TransportKind.WEBRTC: 0.5,
            TransportKind.DOH: 0.2,
        }

    def decide(self, context: BanditContext) -> BanditDecision:
        _ = context
        transport = max(self._scores, key=self._scores.__getitem__)
        return BanditDecision(
            transport=transport,
            score=self._scores[transport],
            model_version=self._model_version,
        )

    def update(self, transport: TransportKind, reward: float) -> None:
        self._scores[transport] = self._scores.get(transport, 0.0) + reward
