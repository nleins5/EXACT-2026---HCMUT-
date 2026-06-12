from __future__ import annotations

import asyncio
import os
import re
import threading
import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.agent.graph import run_pipeline
from src.agent.llm.factory import LLMFactory
from src.api.schemas.request import PredictRequest
from src.api.schemas.response import PredictResult, ReasoningBlock, fallback_response
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


def _constrain_answer_to_options(answer: str, options: list[str]) -> str:
    """If options are provided, force answer to match one of them exactly."""
    if not options:
        return answer

    normalized_answer = answer.strip().casefold()
    aliases = {
        "true": "yes",
        "false": "no",
        "unknown": "uncertain",
        "undetermined": "uncertain",
        "cannot be determined": "uncertain",
    }
    normalized_answer = aliases.get(normalized_answer, normalized_answer)

    for opt in options:
        normalized_option = opt.strip().casefold()
        if aliases.get(normalized_option, normalized_option) == normalized_answer:
            return opt

    # Accept an explicit option marker, but never use substring matching for
    # single-letter options ("answer" contains both A and D).
    marker = re.search(
        r"\b(?:answer|option|choice)\s*(?:is|:|-)?\s*([A-D])\b",
        answer,
        re.IGNORECASE,
    )
    if marker:
        selected = marker.group(1).casefold()
        for opt in options:
            if opt.strip().casefold() == selected:
                return opt

    # Keep the response contract valid without silently changing an uncertain
    # result into "Yes" or "No".
    if normalized_answer in {"unknown", "uncertain"}:
        for opt in options:
            if opt.strip().casefold() in {"unknown", "uncertain"}:
                return opt

    logger.warning(
        "Answer '%s' did not match any option %s; returning the first option as a last resort.",
        answer, options, options[0],
    )
    return options[0]


def _extract_premises_used(result: dict, num_premises: int) -> list[int]:
    """Extract premises_used as 0-based indices from pipeline result."""
    raw = result.get("premises_used")

    # If the pipeline already returns int indices, use them
    if isinstance(raw, list) and raw and all(isinstance(i, int) for i in raw):
        return [i for i in raw if 0 <= i < num_premises]

    # Legacy: premises field contains text — try to match to indices
    premises_text = result.get("premises")
    if isinstance(premises_text, list) and premises_text:
        # If they look like indices already (strings of digits)
        indices = []
        for item in premises_text:
            if isinstance(item, int):
                indices.append(item)
            elif isinstance(item, str) and item.strip().isdigit():
                indices.append(int(item.strip()))
        if indices:
            return [i for i in indices if 0 <= i < num_premises]

    # Fallback for Type 1: if we have premises, assume all were used
    if num_premises > 0 and result.get("task_type") == "logic":
        return list(range(num_premises))

    return []


def _extract_unit(result: dict) -> str:
    """Extract unit for Type 2 responses."""
    unit = result.get("unit")
    if isinstance(unit, str) and unit.strip():
        normalized = (
            unit.strip()
            .replace("μ", "u")
            .replace("µ", "u")
            .replace("Ω", "ohm")
            .replace("²", "^2")
            .replace("³", "^3")
            .replace("°", "degrees ")
            .replace("lần", "times")
        )
        return "" if normalized in {"-", "—"} else normalized
    return ""


def _build_reasoning(result: dict) -> ReasoningBlock | None:
    """Build the reasoning object from pipeline artifacts."""
    fol = _as_text(result.get("fol"))
    cot = result.get("cot")
    steps = []
    reasoning_type = "cot"

    if fol:
        reasoning_type = "fol"
        steps.append(fol)

    if isinstance(cot, list):
        steps.extend([_as_text(s) for s in cot if _as_text(s)])
    elif isinstance(cot, str) and cot.strip():
        steps.append(cot.strip())

    if not steps:
        return None

    return ReasoningBlock(type=reasoning_type, steps=steps)


def _sanitize_response(
    result: dict,
    query_id: str,
    options: list[str],
    num_premises: int,
    task_type_hint: str | None,
) -> PredictResult:
    answer = _as_text(result.get("answer"), default="Unknown")

    if answer.lower() in {"error", ""}:
        if result.get("error_message"):
            logger.warning("Returning fallback — internal: %s", result["error_message"])
        answer = "Unknown"

    explanation = _as_text(result.get("explanation"))
    if not explanation:
        explanation = (
            fallback_response(query_id).explanation
            if answer == "Unknown"
            else "The system returned an answer but did not provide a detailed explanation."
        )

    # Constrain answer to options if provided
    if options:
        answer = _constrain_answer_to_options(answer, options)

    # Determine effective task type
    effective_type = result.get("task_type") or task_type_hint

    # Extract unit (Type 2 only)
    unit = ""
    if effective_type == "physics":
        unit = _extract_unit(result)

    # Extract premises_used (Type 1 only)
    premises_used: list[int] = []
    if effective_type == "logic":
        premises_used = _extract_premises_used(result, num_premises)

    # Build reasoning block
    reasoning = _build_reasoning(result)

    return PredictResult(
        query_id=query_id,
        answer=answer,
        unit=unit,
        explanation=explanation,
        premises_used=premises_used,
        reasoning=reasoning,
    )


def _fallback_for_request(payload: PredictRequest) -> PredictResult:
    return _sanitize_response(
        {
            "task_type": payload.task_type,
            "answer": "Unknown",
            "explanation": fallback_response(payload.query_id).explanation,
        },
        query_id=payload.query_id,
        options=payload.options,
        num_premises=len(payload.premises_nl),
        task_type_hint=payload.task_type,
    )


@router.post("/predict")
async def predict_endpoint(
    payload: PredictRequest,
    request: Request,
) -> JSONResponse:
    """EXACT 2026 prediction endpoint. Returns a JSON list per the Submission Guide."""
    started_at = time.monotonic()
    timeout = _request_timeout()
    cancellation_grace = max(0.5, float(settings.api.cancellation_grace_seconds))
    work_deadline = started_at + max(1.0, timeout - cancellation_grace)
    hard_deadline = started_at + timeout

    question_text = payload.question
    question_preview = question_text[:120].replace("\n", " ")

    logger.info(
        "POST /predict query_id=%s task_type=%s premises=%s options=%s timeout=%ss question=%s",
        payload.query_id,
        payload.task_type or "auto",
        len(payload.premises_nl),
        len(payload.options),
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
                question_text,
                payload.premises_nl,
                task_type=payload.task_type,
                options=payload.options,
                cancel_event=cancel_event,
                deadline=work_deadline,
            )
        )
        result = await asyncio.wait_for(
            asyncio.shield(task),
            timeout=max(0.1, work_deadline - time.monotonic()),
        )
        response = _sanitize_response(
            result,
            query_id=payload.query_id,
            options=payload.options,
            num_premises=len(payload.premises_nl),
            task_type_hint=payload.task_type,
        )
        logger.info(
            "POST /predict completed in %.3fs query_id=%s answer=%s",
            time.monotonic() - started_at,
            payload.query_id,
            response.answer,
        )
        # Return as a JSON list per the Submission Guide §4
        return JSONResponse(content=[response.model_dump()])
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

        fb = _fallback_for_request(payload)
        return JSONResponse(content=[fb.model_dump()])
    except Exception as exc:
        logger.exception("POST /predict failed: %s", exc)
        fb = _fallback_for_request(payload)
        return JSONResponse(content=[fb.model_dump()])
    finally:
        if gate_acquired and release_gate and _predict_gate.locked():
            _predict_gate.release()
