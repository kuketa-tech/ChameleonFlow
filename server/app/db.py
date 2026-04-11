from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import FastAPI
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from server.app.db_models import Base
from server.app.settings import Settings


def build_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(settings.database_url, pool_pre_ping=True)


def build_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def attach_db(app: FastAPI, settings: Settings) -> None:
    engine = build_engine(settings)
    app.state.db_engine = engine
    app.state.db_sessionmaker = build_sessionmaker(engine)


async def get_session(app: FastAPI) -> AsyncIterator[AsyncSession]:
    sessionmaker = app.state.db_sessionmaker
    async with sessionmaker() as session:
        yield session
