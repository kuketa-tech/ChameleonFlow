from __future__ import annotations

from client.app.bandit.service import InMemoryBanditAgent
from shared.contracts import BanditContext, TransportKind


def make_context() -> BanditContext:
    return BanditContext(
        isp_id="isp-a",
        hour_sin=0.0,
        hour_cos=1.0,
        traffic_type="web",
        historical_success_rate=0.5,
    )


def test_bandit_prefers_quic_by_default() -> None:
    agent = InMemoryBanditAgent()

    decision = agent.decide(make_context())

    assert decision.transport == TransportKind.QUIC
    assert decision.model_version == "stub-bandit-v1"


def test_bandit_updates_preference_after_positive_reward() -> None:
    agent = InMemoryBanditAgent()

    agent.update(TransportKind.DOH, reward=1.0)

    decision = agent.decide(make_context())

    assert decision.transport == TransportKind.DOH
