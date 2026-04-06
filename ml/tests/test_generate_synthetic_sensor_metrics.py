from __future__ import annotations

from ml.training.generate_synthetic_sensor_metrics import build_synthetic_sensor_metrics_frame


def test_build_synthetic_sensor_metrics_frame_creates_balanced_labels() -> None:
    frame = build_synthetic_sensor_metrics_frame(sessions=10, rows_per_session=4, seed=42)

    assert len(frame) == 40
    assert set(frame["label"].unique()) == {0, 1}
    assert frame["session_id"].nunique() == 10
