from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from shared.contracts import TransportKind


class HealthResponseSchema(BaseModel):
    status: str


class ModelDescriptorSchema(BaseModel):
    version: str
    artifact_url: str
    signature: str
    created_at: str


class SessionAggregateSchema(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    isp_id: str
    traffic_type: str
    hour_bucket: int
    transport: TransportKind
    success_count: int
    failure_count: int


class AcceptedResponseSchema(BaseModel):
    status: str
