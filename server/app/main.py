from __future__ import annotations

from fastapi import FastAPI

from server.app.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(title="ChameleonFlow Control Server")
    app.include_router(router)
    return app


app = create_app()
