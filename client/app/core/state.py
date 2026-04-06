from __future__ import annotations

from pydantic import BaseModel, Field

from shared.contracts import SessionAggregate, TransportKind


class LocalModelState(BaseModel):
    version: str
    path: str
    signature: str
    activated_at: str | None = None


class ClientState(BaseModel):
    active_transport: TransportKind | None = None
    sensor_model: LocalModelState | None = None
    morpher_model: LocalModelState | None = None
    bandit_model: LocalModelState | None = None
    pending_aggregates: list[SessionAggregate] = Field(default_factory=list)
