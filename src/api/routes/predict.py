from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Any

from fastapi import APIRouter, Request

from src.agent.graph import run_pipeline
from src.agent.llm.factory import LLMFactory
from src.api.schemas.request import PredictRequest
from src.api.schemas.response import PredictResponse, fallback_response
from src.core.config import settings
from src.utils.logger import logger

router = APIRouter(tags=["prediction"])
_predict_gate = asyncio.Lock()


def _request_timeout() -> float:
    raw = os.getenv("EXACT_REQUEST_BUDGET_SECONDS")
    if raw:
        try:
            return max(1.0, float(raw))
        except ValueError:
            logger.warning("Invalid EXACT_REQUEST_BUDGET_SECONDS=%s", raw)
    return float(settings.api.request_budget_seconds)


def _as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        return text or default
    return str(value).strip() or default


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        return [_as_text(item) for item in value if _as_text(item)]
    return [_as_text(value)] if _as_text(value) else []


def _as_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.0
    return min(1.0, max(0.0, confidence))


def _sanitize_response(result: dict) -> PredictResponse:
    answer = _as_text(result.get("answer"), default="Unknown")

    if answer.lower() in {"error", "unknown", ""}:
        if result.get("error_message"):
            logger.warning("Returning fallback — internal: %s", result["error_message"])
        return fallback_response()

    confidence = _as_confidence(result.get("confidence"))
    explanation = _as_text(result.get("explanation"))

    if result.get("code_error"):
        confidence = min(confidence, 0.4)
        logger.info("Solver errored but explanation recovered answer=%s", answer)

    if not explanation:
        explanation = "The system returned an answer but did not provide a detailed explanation."
        confidence = min(confidence, 0.3)

    return PredictResponse(
        answer=answer,
        explanation=explanation,
        fol=_as_text(result.get("fol")),
        cot=_as_text_list(result.get("cot")),
        premises=_as_text_list(result.get("premises")),
        confidence=confidence,
    )


@router.post("/predict", response_model=PredictResponse)
async def predict_endpoint(
    payload: PredictRequest,
    request: Request,
) -> PredictResponse:
    started_at = time.monotonic()
    timeout = _request_timeout()
    cancellation_grace = max(0.5, float(settings.api.cancellation_grace_seconds))
    work_deadline = started_at + max(1.0, timeout - cancellation_grace)
    hard_deadline = started_at + timeout
    question_preview = payload.question[:120].replace("\n", " ")

    logger.info(
        "POST /predict task_type=%s premises=%s timeout=%ss question=%s",
        payload.task_type or "auto",
        len(payload.premises_nl),
        timeout,
        question_preview,
    )

    gate_acquired = False
    release_gate = True
    task: asyncio.Task | None = None
    cancel_event = threading.Event()

    try:
        await asyncio.wait_for(
            _predict_gate.acquire(),
            timeout=max(0.1, work_deadline - time.monotonic()),
        )
        gate_acquired = True

        task = asyncio.create_task(
            asyncio.to_thread(
                run_pipeline,
                payload.question,
                payload.premises_nl,
                task_type=payload.task_type,
                cancel_event=cancel_event,
                deadline=work_deadline,
            )
        )
        result = await asyncio.wait_for(
            asyncio.shield(task),
            timeout=max(0.1, work_deadline - time.monotonic()),
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
        logger.warning("POST /predict exhausted its %.3fs request budget.", timeout)
        cancel_event.set()

        if task is not None:
            logger.warning("Aborting the active pipeline and model process.")
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(LLMFactory.abort_active_request),
                    timeout=max(0.1, hard_deadline - time.monotonic()),
                )
            except asyncio.TimeoutError:
                logger.error("Timed out while aborting the active model process.")

            try:
                await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=max(0.1, hard_deadline - time.monotonic()),
                )
            except Exception:
                pass

            if not task.done():
                release_gate = False

                def release_after_pipeline(done_task: asyncio.Task) -> None:
                    try:
                        done_task.exception()
                    except BaseException:
                        pass
                    if _predict_gate.locked():
                        _predict_gate.release()

                task.add_done_callback(release_after_pipeline)

        return fallback_response()
    except Exception as exc:
        logger.exception("POST /predict failed: %s", exc)
        return fallback_response()
    finally:
        if gate_acquired and release_gate and _predict_gate.locked():
            _predict_gate.release()
