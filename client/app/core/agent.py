
from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone

import aiohttp

from client.app.bandit.service import InMemoryBanditAgent
from client.app.sensor.service import ChannelMetricsWindow, ThresholdSensor
from client.app.transports.registry import build_transport_registry
from shared.contracts import BanditContext, SessionAggregate, TransportKind


@dataclass(slots=True)
class AgentRunResult:
    sessions_attempted: int
    aggregates_sent: int


def _hour_features(now: datetime) -> tuple[float, float, int]:
    hour = now.hour + (now.minute / 60.0)
    angle = 2.0 * math.pi * (hour / 24.0)
    return math.sin(angle), math.cos(angle), int(now.hour)


async def post_aggregate(server_base_url: str, aggregate: SessionAggregate) -> None:
    url = server_base_url.rstrip("/") + "/metrics/aggregates"
    payload = {
        "isp_id": aggregate.isp_id,
        "traffic_type": aggregate.traffic_type,
        "hour_bucket": aggregate.hour_bucket,
        "transport": aggregate.transport,
        "success_count": aggregate.success_count,
        "failure_count": aggregate.failure_count,
    }
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"server rejected aggregate status={resp.status} body={body}")


async def run_client_agent(
    *,
    server_base_url: str,
    isp_id: str = "isp-a",
    traffic_type: str = "web",
    sessions: int = 10,
    seed: int = 42,
) -> AgentRunResult:
    rng = random.Random(seed)
    sensor = ThresholdSensor(threshold=0.7)
    bandit = InMemoryBanditAgent()
    transports = build_transport_registry()

    aggregates_sent = 0

    for _ in range(sessions):
        now = datetime.now(tz=timezone.utc)
        hour_sin, hour_cos, hour_bucket = _hour_features(now)

        context = BanditContext(
            isp_id=isp_id,
            hour_sin=hour_sin,
            hour_cos=hour_cos,
            traffic_type=traffic_type,
            historical_success_rate=0.5,
        )

        decision = bandit.decide(context)
        transport_kind: TransportKind = decision.transport
        transport = transports[transport_kind]

        await transport.connect()

        # MVP: имитируем качество канала и outcome, чтобы прогнать весь контур.
        metrics = ChannelMetricsWindow(
            packet_loss_ratio=rng.random() * 0.4,
            rtt_variation_ratio=rng.random() * 0.6,
            retransmission_ratio=rng.random() * 0.3,
            reset_ratio=rng.random() * 0.1,
        )
        sensor_result = sensor.evaluate(metrics)

        payload = f"chameleonflow:{transport_kind.value}".encode("utf-8")
        await transport.send(payload)
        echoed = await transport.recv()
        await transport.close()

        success = (not sensor_result.degraded) and (echoed == payload)
        reward = 1.0 if success else -1.0
        bandit.update(transport_kind, reward=reward)

        aggregate = SessionAggregate(
            isp_id=isp_id,
            traffic_type=traffic_type,
            hour_bucket=hour_bucket,
            transport=transport_kind,
            success_count=1 if success else 0,
            failure_count=0 if success else 1,
        )

        await post_aggregate(server_base_url, aggregate)
        aggregates_sent += 1

        await asyncio.sleep(0.05)

    return AgentRunResult(sessions_attempted=sessions, aggregates_sent=aggregates_sent)

