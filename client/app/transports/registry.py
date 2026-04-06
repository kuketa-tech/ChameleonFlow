from __future__ import annotations

from client.app.transports.base import TransportPlugin
from client.app.transports.doh import DohTransportPlugin
from client.app.transports.quic import QuicTransportPlugin
from client.app.transports.webrtc import WebRtcTransportPlugin
from shared.contracts import TransportKind


def build_transport_registry() -> dict[TransportKind, TransportPlugin]:
    return {
        TransportKind.DOH: DohTransportPlugin(),
        TransportKind.WEBRTC: WebRtcTransportPlugin(),
        TransportKind.QUIC: QuicTransportPlugin(),
    }
