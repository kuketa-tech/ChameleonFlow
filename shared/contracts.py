from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TransportKind(str, Enum):
    DOH = "doh"
    WEBRTC = "webrtc"
    QUIC = "quic"


@dataclass(slots=True)
class SensorResult:
    probability: float
    degraded: bool
    window_seconds: int = 5


@dataclass(slots=True)
class BanditContext:
    isp_id: str
    hour_sin: float
    hour_cos: float
    traffic_type: str
    historical_success_rate: float


@dataclass(slots=True)
class BanditDecision:
    transport: TransportKind
    score: float
    model_version: str


@dataclass(slots=True)
class SessionAggregate:
    isp_id: str
    traffic_type: str
    hour_bucket: int
    transport: TransportKind
    success_count: int
    failure_count: int
