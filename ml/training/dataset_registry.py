from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = ROOT_DIR / "ml" / "datasets" / "registry.yaml"


class DatasetSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    task: Literal["sensor", "morpher", "shared"]
    title: str
    provider: str
    official_url: str
    local_dir: str
    formats: list[str]
    recommended_for: list[str]
    preprocessing_notes: list[str]

    @property
    def local_path(self) -> Path:
        return ROOT_DIR / self.local_dir


class DatasetRegistry(BaseModel):
    model_config = ConfigDict(frozen=True)

    datasets: list[DatasetSpec]

    def by_task(self, task: Literal["sensor", "morpher"]) -> list[DatasetSpec]:
        return [spec for spec in self.datasets if spec.task in {task, "shared"}]

    def by_id(self, dataset_id: str) -> DatasetSpec:
        for spec in self.datasets:
            if spec.id == dataset_id:
                return spec

        msg = f"Unknown dataset id: {dataset_id}"
        raise KeyError(msg)


def load_dataset_registry(path: Path = DEFAULT_REGISTRY_PATH) -> DatasetRegistry:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return DatasetRegistry.model_validate(raw)
