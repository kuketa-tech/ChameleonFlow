from __future__ import annotations

from client.app.core import agent as agent_module
from shared.contracts import SessionAggregate


async def test_run_client_agent_posts_aggregate_for_each_session(monkeypatch) -> None:
    captured: list[SessionAggregate] = []

    async def fake_post_aggregate(server_base_url: str, aggregate: SessionAggregate) -> None:
        assert server_base_url == "http://testserver"
        captured.append(aggregate)

    monkeypatch.setattr(agent_module, "post_aggregate", fake_post_aggregate)

    result = await agent_module.run_client_agent(
        server_base_url="http://testserver",
        isp_id="isp-test",
        traffic_type="bulk",
        sessions=3,
        seed=7,
    )

    assert result.sessions_attempted == 3
    assert result.aggregates_sent == 3
    assert len(captured) == 3
    assert all(aggregate.isp_id == "isp-test" for aggregate in captured)
