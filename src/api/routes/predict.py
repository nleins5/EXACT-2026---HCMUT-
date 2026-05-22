from __future__ import annotations

import asyncio
import os
import time

from fastapi import APIRouter, Request

from src.agent.graph import run_pipeline
from src.api.schemas.request import PredictRequest
from src.api.schemas.response import PredictResponse, fallback_response
from src.core.config import settings
from src.utils.logger import logger

router = APIRouter(tags=["prediction"])


def _request_timeout() -> float:
    raw = os.getenv("EXACT_REQUEST_BUDGET_SECONDS")
    if raw:
        try:
            return max(1.0, float(raw))
        except ValueError:
            logger.warning("Invalid EXACT_REQUEST_BUDGET_SECONDS=%s", raw)
    return float(settings.api.request_budget_seconds)


def _sanitize_response(result: dict) -> PredictResponse:
    answer = str(result.get("answer") or "Unknown")
    if answer.lower() == "error" or result.get("code_error"):
        return fallback_response(result.get("error_message") or None)

    return PredictResponse(
        answer=answer,
        explanation=str(result.get("explanation") or ""),
        fol=str(result.get("fol") or ""),
        cot=result.get("cot") or [],
        premises=result.get("premises") or [],
        confidence=float(result.get("confidence") or 0.0),
    )


@router.post("/predict", response_model=PredictResponse)
async def predict_endpoint(
    payload: PredictRequest,
    request: Request,
) -> PredictResponse:
    started_at = time.monotonic()
    timeout = _request_timeout()
    question_preview = payload.question[:120].replace("\n", " ")

    logger.info(
        "POST /predict premises=%s timeout=%ss question=%s",
        len(payload.premises_nl),
        timeout,
        question_preview,
    )

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                run_pipeline,
                payload.question,
                payload.premises_nl,
            ),
            timeout=timeout,
        )
        response = _sanitize_response(result)
        logger.info(
            "POST /predict completed in %.3fs answer=%s confidence=%.3f",
            time.monotonic() - started_at,
            response.answer,
            response.confidence,
        )
        return response
    except asyncio.TimeoutError:
        logger.warning("POST /predict timed out after %.3fs", timeout)
        return fallback_response("timeout")
    except Exception as exc:
        logger.exception("POST /predict failed")
        if getattr(request.app.state, "debug", False):
            return fallback_response(str(exc))
        return fallback_response()
