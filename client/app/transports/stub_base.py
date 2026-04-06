from __future__ import annotations

import asyncio

from client.app.transports.base import TransportPlugin
from shared.contracts import TransportKind


class StubTransportPlugin(TransportPlugin):
    def __init__(self, kind: TransportKind) -> None:
        self.kind = kind
        self._connected = False
        self._recv_queue: asyncio.Queue[bytes] = asyncio.Queue()

    async def connect(self) -> None:
        self._connected = True

    async def send(self, payload: bytes) -> None:
        if not self._connected:
            msg = f"{self.kind.value} transport is not connected"
            raise RuntimeError(msg)

        await self._recv_queue.put(payload)

    async def recv(self) -> bytes:
        if not self._connected:
            msg = f"{self.kind.value} transport is not connected"
            raise RuntimeError(msg)

        return await self._recv_queue.get()

    async def close(self) -> None:
        self._connected = False

    async def health(self) -> bool:
        return self._connected
