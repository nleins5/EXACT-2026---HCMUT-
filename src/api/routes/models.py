"""Model metadata endpoint with cache fallback during model swap."""
from __future__ import annotations

import json
import threading

import httpx
from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse

from src.core.config import settings

router = APIRouter(tags=["model"])


_cache_lock = threading.Lock()
_cached_response: bytes | None = None


def _configured_models_payload(error: str | None = None) -> dict:
    """OpenAI-style model list from config."""

    def one(role: str, cfg) -> dict:
        return {
            "id": cfg.model_name,
            "object": "model",
            "created": 0,
            "owned_by": "exact-2026",
            "root": cfg.model_name,
            "parent": None,
            "metadata": {
                "role": role,
                "parameter_class": "7B-class",
                "serving": "self-hosted OpenAI-compatible llama-server",
                "model_path": cfg.model_path,
            },
        }

    payload = {
        "object": "list",
        "data": [
            one("coder", settings.llm.coder),
            one("instruct", settings.llm.instruct),
        ],
        "exact_runtime": {
            "single_resident": True,
            "base_url": settings.llm.server.base_url,
            "note": "LlamaServerSupervisor keeps at most one LLM process resident.",
        },
    }
    if error:
        payload["error"] = error
    return payload


@router.get("/v1/models")
async def models_endpoint() -> Response:
    """Expose OpenAI-compatible model metadata."""
    global _cached_response

    models_url = f"{settings.llm.server.base_url.rstrip('/')}/models"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(models_url)


        if response.status_code != 200:
            return JSONResponse(
                status_code=503,
                content=_configured_models_payload(
                    f"llama-server returned HTTP {response.status_code}"
                ),
            )

        try:
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("upstream /models returned a non-object payload")
        except ValueError:
            return JSONResponse(
                status_code=503,
                content=_configured_models_payload("llama-server returned invalid JSON"),
            )

        encoded = json.dumps(payload).encode("utf-8")
        with _cache_lock:
            _cached_response = encoded

        return JSONResponse(
            status_code=200,
            content=payload,
        )

    except httpx.HTTPError as exc:

        with _cache_lock:
            cached = _cached_response

        if cached is not None:
            try:
                data = json.loads(cached)
                data["swap_in_progress"] = True
                return JSONResponse(
                    status_code=200,
                    content=data,
                )
            except (json.JSONDecodeError, TypeError):
                pass

        # No cache yet: expose configured model metadata
        return JSONResponse(
            status_code=503,
            content=_configured_models_payload(f"llama-server is not reachable: {exc}"),
        )
