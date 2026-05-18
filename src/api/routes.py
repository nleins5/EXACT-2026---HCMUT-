"""
Inference route for the EXACT 2026 HTTP API.

Single endpoint: ``POST /predict`` — accepts the BTC payload, runs the
LangGraph pipeline, returns the structured response.

The per-request budget is configurable via the ``EXACT_REQUEST_BUDGET_SECONDS``
environment variable. The competition limit is 60 seconds (slide 13), but for
development on CPU-only hardware we default to 600s so the pipeline actually
finishes. Set the env var to ``58`` before deploying to a competition runner.
"""
from __future__ import annotations

import asyncio
import os
import time

from fastapi import APIRouter, HTTPException, Request

from src.agent.graph import run_pipeline
from src.api.schemas import PredictRequest, PredictResponse
from src.utils.logger import logger

router = APIRouter(tags=["inference"])

# Competition spec (slide 13) caps requests at 60s. Default to 600s for local
# CPU dev so we can demo the full pipeline; override for prod.
_REQUEST_BUDGET_SECONDS = float(os.getenv("EXACT_REQUEST_BUDGET_SECONDS", "600"))


@router.post(
    "/predict",
    response_model=PredictResponse,
    response_model_exclude_none=False,
    summary="Run a single EXACT 2026 query through the agent pipeline.",
)
async def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    """Run inference on a single query.

    Args:
        payload:  Validated request body. ``premises-NL`` is optional and
            distinguishes Type 1 (logic) from Type 2 (physics).
        request:  Underlying FastAPI request, unused for now but kept for
            future per-request tracing / cancellation hooks.

    Returns:
        :class:`PredictResponse` matching the BTC schema.

    Raises:
        HTTPException: 504 on timeout, 500 on unexpected pipeline failure.
    """
    t0 = time.perf_counter()
    premises = payload.premises_nl or []
    logger.info(
        f"[/predict] received | type={'logic' if premises else 'physics'} | "
        f"q={payload.question[:80]!r} | n_premises={len(premises)}"
    )

    loop = asyncio.get_running_loop()
    try:
        # ``run_pipeline`` is sync (LangGraph + llama.cpp); offload to executor
        # so the event loop stays responsive and we can apply a hard timeout.
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: run_pipeline(question=payload.question, premises=premises),
            ),
            timeout=_REQUEST_BUDGET_SECONDS,
        )
    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - t0
        logger.error(f"[/predict] TIMEOUT after {elapsed:.1f}s (budget={_REQUEST_BUDGET_SECONDS}s)")
        raise HTTPException(
            status_code=504,
            detail=f"Inference exceeded {_REQUEST_BUDGET_SECONDS}s budget.",
        )
    except Exception as exc:  # pragma: no cover — defensive, logged below
        elapsed = time.perf_counter() - t0
        logger.exception(f"[/predict] FAILED after {elapsed:.1f}s: {exc}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    elapsed = time.perf_counter() - t0
    logger.info(
        f"[/predict] done in {elapsed:.1f}s | answer={result.get('answer')!r} | "
        f"task_type={result.get('task_type')}"
    )

    # ``run_pipeline`` may legitimately return None for optional fields; let
    # PredictResponse drop them rather than emitting empty strings/lists.
    return PredictResponse(
        answer=result.get("answer") or "",
        explanation=result.get("explanation") or "",
        fol=result.get("fol") or None,
        cot=result.get("cot") or None,
        premises=result.get("premises") or None,
        confidence=result.get("confidence"),
    )
