from __future__ import annotations

import time

from fastapi import APIRouter, Request

from src.api.schemas.response import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_endpoint(request: Request) -> HealthResponse:
    started_at = getattr(request.app.state, "started_at", time.monotonic())
    supervisor = getattr(request.app.state, "supervisor", None)

    return HealthResponse(
        status="ok",
        supervisor_running=bool(supervisor and supervisor.is_alive()),
        active_role=getattr(supervisor, "active_role", None),
        uptime=round(time.monotonic() - started_at, 3),
    )

