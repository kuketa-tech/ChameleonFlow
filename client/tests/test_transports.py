from __future__ import annotations

import pytest

from client.app.transports.registry import build_transport_registry
from shared.contracts import TransportKind


@pytest.mark.asyncio
async def test_transport_registry_contains_all_declared_transports() -> None:
    registry = build_transport_registry()

    assert set(registry) == {
        TransportKind.DOH,
        TransportKind.WEBRTC,
        TransportKind.QUIC,
    }


@pytest.mark.asyncio
async def test_quic_stub_transport_lifecycle() -> None:
    registry = build_transport_registry()
    transport = registry[TransportKind.QUIC]

    assert await transport.health() is False

    await transport.connect()
    assert await transport.health() is True

    await transport.send(b"hello")
    received = await transport.recv()

    assert received == b"hello"

    await transport.close()
    assert await transport.health() is False


@pytest.mark.asyncio
async def test_send_without_connect_raises_runtime_error() -> None:
    registry = build_transport_registry()
    transport = registry[TransportKind.DOH]

    with pytest.raises(RuntimeError, match="not connected"):
        await transport.send(b"payload")
