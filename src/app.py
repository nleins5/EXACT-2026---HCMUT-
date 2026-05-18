"""FastAPI app — long-running middleware cho EXACT 2026.

Lifespan:
- Khoi tao LlamaServerSupervisor.
- swap_to("instruct") (warm-up). Lan goi formalizer dau tien moi swap sang coder.
- Cleanup: shutdown supervisor (kill llama-server).

Run:
    uvicorn src.app:app --host 0.0.0.0 --port 8000 --workers 1

NOTE: Luon dung --workers 1. Moi worker se spawn rieng llama-server -> OOM.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.agent.llm.factory import LLMFactory
from src.agent.llm.server_supervisor import LlamaServerSupervisor
from src.api.routes import router as inference_router
from src.api.schemas import HealthResponse
from src.utils.logger import logger


# Lifespan se cap nhat — /health doc tu day.
_warmup_state: dict = {"ready": False, "elapsed_seconds": None, "supervisor": None}


def _warmup() -> tuple[float, LlamaServerSupervisor]:
    """Spawn llama-server lan dau (role = instruct).

    Lua chon role mac dinh = instruct vi:
    - Trong dataset, ~50% bai chua co premises (rieng physics) -> classifier
      route physics_rag -> formalizer (coder swap toi day).
    - Logic flow co phan classify -> formalizer (coder) -> solver -> explanation
      (instruct). Warm-up instruct nghia la lan dau sap RA explanation se nhanh.
    - Voi cap nhat sau, co the chuyen sang warm coder neu workload thay doi.
    """
    t0 = time.perf_counter()
    logger.info("Warming up llama-server (role=instruct)...")

    supervisor = LlamaServerSupervisor()
    LLMFactory.init(supervisor)
    supervisor.swap_to("instruct")

    elapsed = time.perf_counter() - t0
    logger.info(f"llama-server ready in {elapsed:.1f}s.")
    return elapsed, supervisor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifecycle: warm model truoc khi serve traffic, cleanup khi shutdown."""
    try:
        elapsed, supervisor = _warmup()
        _warmup_state["ready"] = True
        _warmup_state["elapsed_seconds"] = elapsed
        _warmup_state["supervisor"] = supervisor
    except Exception as exc:
        logger.exception(f"FATAL: warm-up llama-server that bai: {exc}")
        # Khong raise — de /health bao not-ready, operator van vao container debug duoc.
        _warmup_state["ready"] = False
        _warmup_state["elapsed_seconds"] = None
        _warmup_state["supervisor"] = None

    yield

    logger.info("Shutting down EXACT 2026 service.")
    sup = _warmup_state.get("supervisor")
    if sup is not None:
        try:
            sup.shutdown()
        except Exception as exc:
            logger.warning(f"Supervisor shutdown loi: {exc}")


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

# CORS thoang — eval system goi cross-origin. Tighten khi deploy production.
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
