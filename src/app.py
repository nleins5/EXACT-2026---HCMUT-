"""
FastAPI application entry-point for EXACT 2026.

Designed as a long-running middleware:

* On startup, the GGUF model is loaded **once** via the lifespan handler.
  This pulls the 60-second model load out of the per-request budget.
* On every subsequent request, ``LLMFactory.create_client`` hits the
  in-process cache and reuses the same ``ChatLlamaCpp`` instance, so the
  pipeline only spends time on inference.

Run locally:
    uvicorn src.app:app --host 0.0.0.0 --port 8000 --workers 1

> NOTE: Always run with ``--workers 1``. Each worker would load its own
> 5GB GGUF copy into RAM, which will OOM on most laptops.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router as inference_router
from src.api.schemas import HealthResponse
from src.llm.factory import LLMFactory
from src.utils.logger import logger


# Filled in by the lifespan handler so /health can report cold-start cost.
_warmup_state: dict = {"ready": False, "elapsed_seconds": None}


def _warmup_model() -> float:
    """Pre-load the GGUF model into RAM.

    Done synchronously at startup so the first request doesn't pay the
    one-off 30-60s mmap + tokenizer init cost.

    Returns:
        Elapsed seconds spent warming up.
    """
    t0 = time.perf_counter()
    logger.info("Warming up LLM (this loads the GGUF into memory)...")

    # ``reasoning`` matches the purpose used by formalizer/explanation nodes,
    # so they will reuse this exact cached client (same path + temperature).
    client = LLMFactory.create_client(purpose="reasoning")
    _ = client.get_llm()

    elapsed = time.perf_counter() - t0
    logger.info(f"LLM warm-up done in {elapsed:.1f}s.")
    return elapsed


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: warm the model once before serving traffic."""
    try:
        elapsed = _warmup_model()
        _warmup_state["ready"] = True
        _warmup_state["elapsed_seconds"] = elapsed
    except Exception as exc:  # pragma: no cover — surfaced to operator logs
        logger.exception(f"FATAL: model warm-up failed: {exc}")
        # Do NOT raise — let /health report not-ready instead of crashing the
        # whole server, so the operator can still exec into the container.
        _warmup_state["ready"] = False
        _warmup_state["elapsed_seconds"] = None

    yield

    logger.info("Shutting down EXACT 2026 service.")


app = FastAPI(
    title="EXACT 2026 Inference Service",
    version="0.1.0",
    description=(
        "Neuro-symbolic agent for the HCMUT EXACT 2026 challenge. "
        "Routes Type 1 (logic) queries through Z3 and Type 2 (physics) "
        "through SymPy."
    ),
    lifespan=lifespan,
)

# Permissive CORS — the eval system calls us cross-origin. Tighten for prod
# behind a reverse proxy if needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    """Liveness + readiness probe."""
    return HealthResponse(
        status="ok" if _warmup_state["ready"] else "warming",
        model_loaded=_warmup_state["ready"],
        elapsed_warmup_seconds=_warmup_state["elapsed_seconds"],
    )


app.include_router(inference_router)
