from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHAMELEONFLOW_", extra="ignore")

    env: str = "development"
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    database_url: str = "postgresql+asyncpg://chameleonflow:chameleonflow@localhost:5432/chameleonflow"
    models_dir: str = "./models"


def load_settings() -> Settings:
    return Settings()

