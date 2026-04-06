from __future__ import annotations

from ml.training.dataset_registry import load_dataset_registry


def test_dataset_registry_exposes_shared_vpn_dataset_for_both_tasks() -> None:
    registry = load_dataset_registry()

    sensor_ids = {spec.id for spec in registry.by_task("sensor")}
    morpher_ids = {spec.id for spec in registry.by_task("morpher")}

    assert "iscxvpn2016" in sensor_ids
    assert "iscxvpn2016" in morpher_ids


def test_dataset_registry_returns_expected_spec() -> None:
    registry = load_dataset_registry()
    spec = registry.by_id("iscxvpn2016")

    assert spec.title == "ISCX VPN-nonVPN 2016"
    assert spec.local_path.as_posix().endswith("ml/datasets/raw/iscxvpn2016")
