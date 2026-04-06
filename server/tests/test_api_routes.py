from __future__ import annotations

from httpx import ASGITransport, AsyncClient

from server.app.main import app


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
