from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from src.agent.llm.factory import LLMFactory
from src.agent.llm.server_supervisor import LlamaServerSupervisor
from src.api.routes import health, models, predict
from src.core.config import settings
from src.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.started_at = time.monotonic()
    app.state.debug = settings.app.debug
    app.state.supervisor = LlamaServerSupervisor()
    LLMFactory.init(app.state.supervisor)
    logger.info("EXACT API startup complete.")
    try:
        yield
    finally:
        app.state.supervisor.shutdown()
        LLMFactory.reset()
        logger.info("EXACT API shutdown complete.")


app = FastAPI(
    title=settings.app.project_name,
    version=settings.app.version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(models.router)
app.include_router(predict.router)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")
