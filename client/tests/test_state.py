from __future__ import annotations

from client.app.core.state import ClientState, LocalModelState


def test_client_state_roundtrip() -> None:
    state = ClientState(
        active_transport="quic",
        sensor_model=LocalModelState(
            version="sensor-v1",
            path="ml/exported/sensor.onnx",
            signature="deadbeef",
            activated_at="2026-04-06T10:00:00Z",
        ),
        pending_aggregates=[],
    )

    restored = ClientState.model_validate(state.model_dump(mode="python"))

    assert restored.active_transport == "quic"
    assert restored.sensor_model is not None
    assert restored.sensor_model.version == "sensor-v1"
