from __future__ import annotations

import httpx
from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse

from src.core.config import settings

router = APIRouter(tags=["model"])


@router.get("/v1/models")
async def models_endpoint() -> Response:
    """Expose OpenAI-compatible model metadata for EXACT verification."""

    models_url = f"{settings.llm.server.base_url.rstrip('/')}/models"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(models_url)
    except httpx.HTTPError as exc:
        return JSONResponse(
            status_code=503,
            content={
                "object": "list",
                "data": [],
                "error": f"llama-server is not reachable: {exc}",
            },
        )

    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=response.headers.get("content-type", "application/json"),
    )

