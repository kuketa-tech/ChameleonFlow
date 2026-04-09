from __future__ import annotations

from fastapi import APIRouter, Request
from sqlalchemy import desc, select

from shared.contracts import SessionAggregate
from server.app.db_models import SessionAggregateRow
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
async def ingest_aggregate(request: Request, aggregate: SessionAggregateSchema) -> AcceptedResponseSchema:
    domain_aggregate = SessionAggregate(
        isp_id=aggregate.isp_id,
        traffic_type=aggregate.traffic_type,
        hour_bucket=aggregate.hour_bucket,
        transport=aggregate.transport,
        success_count=aggregate.success_count,
        failure_count=aggregate.failure_count,
    )
    sessionmaker = request.app.state.db_sessionmaker
    async with sessionmaker() as session:
        session.add(
            SessionAggregateRow(
                isp_id=domain_aggregate.isp_id,
                traffic_type=domain_aggregate.traffic_type,
                hour_bucket=domain_aggregate.hour_bucket,
                transport=domain_aggregate.transport,
                success_count=domain_aggregate.success_count,
                failure_count=domain_aggregate.failure_count,
            )
        )
        await session.commit()
    return AcceptedResponseSchema(status="accepted")


@router.get("/metrics/aggregates/recent")
async def list_recent_aggregates(request: Request, limit: int = 20) -> list[SessionAggregateSchema]:
    limit = max(1, min(limit, 200))
    sessionmaker = request.app.state.db_sessionmaker
    async with sessionmaker() as session:
        result = await session.execute(
            select(SessionAggregateRow).order_by(desc(SessionAggregateRow.id)).limit(limit)
        )
        rows = list(result.scalars().all())
    return [
        SessionAggregateSchema(
            isp_id=row.isp_id,
            traffic_type=row.traffic_type,
            hour_bucket=row.hour_bucket,
            transport=row.transport,
            success_count=row.success_count,
            failure_count=row.failure_count,
        )
        for row in rows
    ]
