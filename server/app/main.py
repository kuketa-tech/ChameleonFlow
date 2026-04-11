from __future__ import annotations

from fastapi import FastAPI

from server.app.api.routes import router
from server.app.db import attach_db, init_db
from server.app.settings import load_settings


def create_app() -> FastAPI:
    settings = load_settings()
    app = FastAPI(title="ChameleonFlow Control Server")
    attach_db(app, settings)
    app.include_router(router)

    @app.on_event("startup")
    async def _startup() -> None:
        await init_db(app.state.db_engine)

    return app


app = create_app()
