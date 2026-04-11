from __future__ import annotations

from dataclasses import dataclass

from httpx import ASGITransport, AsyncClient

from shared.contracts import TransportKind
from server.app.db_models import SessionAggregateRow
from server.app.main import app


class FakeScalarResult:
    def __init__(self, rows: list[SessionAggregateRow]) -> None:
        self._rows = rows

    def all(self) -> list[SessionAggregateRow]:
        return self._rows


class FakeExecuteResult:
    def __init__(self, rows: list[SessionAggregateRow]) -> None:
        self._rows = rows

    def scalars(self) -> FakeScalarResult:
        return FakeScalarResult(self._rows)


@dataclass
class FakeSession:
    rows: list[SessionAggregateRow]
    added: list[object]
    committed: bool = False

    async def __aenter__(self) -> FakeSession:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def add(self, row: object) -> None:
        self.added.append(row)

    async def commit(self) -> None:
        self.committed = True

    async def execute(self, statement: object) -> FakeExecuteResult:
        _ = statement
        return FakeExecuteResult(self.rows)


class FakeSessionMaker:
    def __init__(self, rows: list[SessionAggregateRow] | None = None) -> None:
        self._rows = rows or []
        self.sessions: list[FakeSession] = []

    def __call__(self) -> FakeSession:
        session = FakeSession(rows=self._rows, added=[])
        self.sessions.append(session)
        return session


async def test_health_endpoint_returns_ok() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_latest_model_endpoint_returns_stub_descriptor() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/models/latest")

    assert response.status_code == 200
    assert response.json() == {
        "version": "stub",
        "artifact_url": "/artifacts/stub",
        "signature": "stub",
        "created_at": "1970-01-01T00:00:00Z",
    }


async def test_aggregate_endpoint_accepts_session_aggregate() -> None:
    sessionmaker = FakeSessionMaker()
    app.state.db_sessionmaker = sessionmaker

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/metrics/aggregates",
            json={
                "isp_id": "isp-a",
                "traffic_type": "web",
                "hour_bucket": 12,
                "transport": "quic",
                "success_count": 3,
                "failure_count": 1,
            },
        )

    assert response.status_code == 200
    assert response.json() == {"status": "accepted"}
    assert len(sessionmaker.sessions) == 1
    assert sessionmaker.sessions[0].committed is True
    inserted = sessionmaker.sessions[0].added[0]
    assert inserted.isp_id == "isp-a"
    assert inserted.success_count == 3
    assert inserted.failure_count == 1


async def test_recent_aggregates_endpoint_returns_rows() -> None:
    sessionmaker = FakeSessionMaker(
        rows=[
            SessionAggregateRow(
                isp_id="isp-a",
                traffic_type="web",
                hour_bucket=12,
                transport=TransportKind.QUIC,
                success_count=3,
                failure_count=1,
            ),
            SessionAggregateRow(
                isp_id="isp-b",
                traffic_type="video",
                hour_bucket=8,
                transport=TransportKind.WEBRTC,
                success_count=5,
                failure_count=0,
            ),
        ]
    )
    app.state.db_sessionmaker = sessionmaker

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/metrics/aggregates/recent?limit=2")

    assert response.status_code == 200
    assert response.json() == [
        {
            "isp_id": "isp-a",
            "traffic_type": "web",
            "hour_bucket": 12,
            "transport": "quic",
            "success_count": 3,
            "failure_count": 1,
        },
        {
            "isp_id": "isp-b",
            "traffic_type": "video",
            "hour_bucket": 8,
            "transport": "webrtc",
            "success_count": 5,
            "failure_count": 0,
        },
    ]
