from __future__ import annotations

from fastapi import APIRouter

from shared.contracts import SessionAggregate
from server.app.api.schemas import (
    AcceptedResponseSchema,
    HealthResponseSchema,
    ModelDescriptorSchema,
    SessionAggregateSchema,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponseSchema)
async def health() -> HealthResponseSchema:
    return HealthResponseSchema(status="ok")


@router.get("/models/latest", response_model=ModelDescriptorSchema)
async def get_latest_model() -> ModelDescriptorSchema:
    return ModelDescriptorSchema(
        version="stub",
        artifact_url="/artifacts/stub",
        signature="stub",
        created_at="1970-01-01T00:00:00Z",
    )


@router.post("/metrics/aggregates", response_model=AcceptedResponseSchema)
async def ingest_aggregate(aggregate: SessionAggregateSchema) -> AcceptedResponseSchema:
    domain_aggregate = SessionAggregate(
        isp_id=aggregate.isp_id,
        traffic_type=aggregate.traffic_type,
        hour_bucket=aggregate.hour_bucket,
        transport=aggregate.transport,
        success_count=aggregate.success_count,
        failure_count=aggregate.failure_count,
    )
    _ = domain_aggregate
    return AcceptedResponseSchema(status="accepted")
