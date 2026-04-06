from __future__ import annotations

from client.app.transports.stub_base import StubTransportPlugin
from shared.contracts import TransportKind


class WebRtcTransportPlugin(StubTransportPlugin):
    def __init__(self) -> None:
        super().__init__(TransportKind.WEBRTC)
