from __future__ import annotations

from abc import ABC, abstractmethod

from shared.contracts import TransportKind


class TransportPlugin(ABC):
    kind: TransportKind

    @abstractmethod
    async def connect(self) -> None:
        """Open the transport session."""

    @abstractmethod
    async def send(self, payload: bytes) -> None:
        """Send an opaque payload chunk."""

    @abstractmethod
    async def recv(self) -> bytes:
        """Receive an opaque payload chunk."""

    @abstractmethod
    async def close(self) -> None:
        """Close transport resources."""

    @abstractmethod
    async def health(self) -> bool:
        """Return current transport health status."""
