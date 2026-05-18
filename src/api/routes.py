"""Inference route cho EXACT 2026 HTTP API.

Single endpoint: ``POST /predict`` — accept BTC payload, run LangGraph pipeline,
tra ket qua structured.

Budget per-request lay tu settings.api.request_budget_seconds. BTC Q13 cap 60s;
default = 600 cho dev (override bang env EXACT_REQUEST_BUDGET_SECONDS).
"""
from __future__ import annotations

import asyncio
import os
import time

from fastapi import APIRouter, HTTPException, Request

from src.agent.graph import run_pipeline
from src.api.schemas import PredictRequest, PredictResponse
from src.core.config import settings
from src.utils.logger import logger

router = APIRouter(tags=["inference"])


def _resolve_budget() -> float:
    """Doc budget tu env (uu tien) hoac settings.api.request_budget_seconds."""
    env_v = os.getenv("EXACT_REQUEST_BUDGET_SECONDS")
    if env_v:
        try:
            return float(env_v)
        except ValueError:
            logger.warning(
                f"EXACT_REQUEST_BUDGET_SECONDS khong phai so: {env_v!r}, "
                f"dung settings.api.request_budget_seconds."
            )
    return float(settings.api.request_budget_seconds)


@router.post(
    "/predict",
    response_model=PredictResponse,
    response_model_exclude_none=False,
    summary="Run a single EXACT 2026 query through the agent pipeline.",
)
async def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    """Run inference cho 1 query.

    Args:
        payload: validated request body. ``premises-NL`` optional, distinguishes
                 Type 1 (logic) vs Type 2 (physics).
        request: FastAPI request, reserved cho per-request tracing tuong lai.

    Returns:
        PredictResponse khop schema BTC.

    Raises:
        HTTPException: 504 timeout, 500 unexpected pipeline failure.
    """
    budget = _resolve_budget()
    t0 = time.perf_counter()
    premises = payload.premises_nl or []
    logger.info(
        f"[/predict] received | type={'logic' if premises else 'physics'} | "
        f"q={payload.question[:80]!r} | n_premises={len(premises)}"
    )

    loop = asyncio.get_running_loop()
    try:
        # run_pipeline la sync (LangGraph + LLM HTTP); offload sang executor de
        # event loop con responsive va apply duoc hard timeout.
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: run_pipeline(question=payload.question, premises=premises),
            ),
            timeout=budget,
        )
    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - t0
        logger.error(f"[/predict] TIMEOUT after {elapsed:.1f}s (budget={budget}s)")
        raise HTTPException(
            status_code=504,
            detail=f"Inference exceeded {budget}s budget.",
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.exception(f"[/predict] FAILED after {elapsed:.1f}s: {exc}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    elapsed = time.perf_counter() - t0
    logger.info(
        f"[/predict] done in {elapsed:.1f}s | answer={result.get('answer')!r} | "
        f"task_type={result.get('task_type')}"
    )

    return PredictResponse(
        answer=result.get("answer") or "",
        explanation=result.get("explanation") or "",
        fol=result.get("fol") or None,
        cot=result.get("cot") or None,
        premises=result.get("premises") or None,
        confidence=result.get("confidence"),
    )
