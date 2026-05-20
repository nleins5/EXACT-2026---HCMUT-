"""Teacher LLM client - async wrapper for Gemini (default) or OpenAI.

Responsibilities:
- Call teacher LLM with JSON-mode + few-shot examples.
- Parse output, validate schema.
- Retry exponential backoff for rate-limit / 5xx.
- Return KBRecord (not yet verified).

Concurrency control is handled by caller (asyncio.Semaphore).
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from scripts.distill.prompts import (
    EXTRACT_FEW_SHOT_ASSISTANT_1,
    EXTRACT_FEW_SHOT_ASSISTANT_2,
    EXTRACT_FEW_SHOT_USER_1,
    EXTRACT_FEW_SHOT_USER_2,
    EXTRACT_SYSTEM_PROMPT,
    GENERATE_FEW_SHOT_ASSISTANT_1,
    GENERATE_FEW_SHOT_ASSISTANT_2,
    GENERATE_FEW_SHOT_USER_1,
    GENERATE_FEW_SHOT_USER_2,
    GENERATE_SYSTEM_PROMPT,
    build_extract_user_prompt,
    build_generate_user_prompt,
)
from scripts.distill.schema import KBRecord
from src.utils.logger import logger


class TeacherClientError(Exception):
    """Raised when teacher LLM returns invalid output after all retries."""


class GeminiTeacherClient:
    """Async client for Gemini API (google-generativeai SDK).

    Gemini has no dedicated 'system' role -> inject system into history.
    Force JSON output via response_mime_type=application/json.
    """

    def __init__(self) -> None:
        from src.core.config import settings
        cfg = settings.distillation.teacher
        self.model_name = cfg.model_name
        self.temperature = cfg.temperature
        self.max_output_tokens = cfg.max_output_tokens
        self.timeout_s = settings.distillation.pipeline.timeout_s
        self.max_retries = settings.distillation.pipeline.max_retries
        self.retry_backoff_s = settings.distillation.pipeline.retry_backoff_s
        self.mode = settings.distillation.pipeline.mode  # extract | generate

        api_key = os.environ.get(cfg.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing {cfg.api_key_env} in environment. "
                f"Set it in .env or shell before running distillation."
            )

        # Lazy-import so runtime doesn't depend on google-generativeai.
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self._genai = genai
        # Don't keep a reference to _model -> create new model each turn so
        # generation_config and system_instruction are always correct.

    async def _call_once(
        self,
        problem: str,
        hint: str,
        answer: str = "",
        unit: str = "",
    ) -> dict[str, Any]:
        """Single call to Gemini, raise on schema/JSON error.

        Mode `extract` (default):
          - Requires `hint` (CoT) + `answer` (ground truth).
          - Falls back to `generate` when hint is missing.
        Mode `generate`:
          - Reason from `problem` + optional `hint`.
        """
        use_extract = self.mode == "extract" and bool(hint.strip()) and bool(answer.strip())

        if use_extract:
            system_prompt = EXTRACT_SYSTEM_PROMPT
            history = [
                {"role": "user", "parts": [EXTRACT_FEW_SHOT_USER_1]},
                {"role": "model", "parts": [EXTRACT_FEW_SHOT_ASSISTANT_1]},
                {"role": "user", "parts": [EXTRACT_FEW_SHOT_USER_2]},
                {"role": "model", "parts": [EXTRACT_FEW_SHOT_ASSISTANT_2]},
            ]
            user_msg = build_extract_user_prompt(problem, hint, answer, unit)
        else:
            system_prompt = GENERATE_SYSTEM_PROMPT
            history = [
                {"role": "user", "parts": [GENERATE_FEW_SHOT_USER_1]},
                {"role": "model", "parts": [GENERATE_FEW_SHOT_ASSISTANT_1]},
                {"role": "user", "parts": [GENERATE_FEW_SHOT_USER_2]},
                {"role": "model", "parts": [GENERATE_FEW_SHOT_ASSISTANT_2]},
            ]
            user_msg = build_generate_user_prompt(problem, hint)

        # Gemini SDK: system_instruction is separate in GenerativeModel ctor.
        # response_mime_type=application/json forces pure JSON output -> parse directly.
        model = self._genai.GenerativeModel(
            self.model_name,
            system_instruction=system_prompt,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
                "response_mime_type": "application/json",
            },
        )
        chat = model.start_chat(history=history)

        # SDK has no native async -> run in threadpool.
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: chat.send_message(user_msg)),
            timeout=self.timeout_s,
        )

        # Token usage
        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
        output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0

        text = response.text or ""
        # Gemini with response_mime_type=json returns pure JSON, parse directly.
        parsed = json.loads(text)
        parsed["_input_tokens"] = input_tokens
        parsed["_output_tokens"] = output_tokens
        parsed["_mode"] = "extract" if use_extract else "generate"
        return parsed

    async def distill_one(
        self,
        *,
        record_id: str,
        source: str,
        problem: str,
        hint: str = "",
        answer: str = "",
        unit: str = "",
    ) -> KBRecord:
        """Distill 1 problem -> KBRecord. Auto-retry on transient errors.

        Args:
            record_id, source, problem: required.
            hint: CoT solution available (extract mode).
            answer: ground-truth final answer (extract mode).
            unit: SI unit of answer (optional).
        """
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                data = await self._call_once(problem, hint, answer, unit)
                return KBRecord(
                    id=record_id,
                    source=source,
                    problem=problem,
                    topic=data.get("topic", "other"),
                    formulas=data.get("formulas", []) or [],
                    symbols=data.get("symbols", {}) or {},
                    sympy_code=data.get("sympy_code", ""),
                    answer=data.get("answer", answer),
                    derivation=data.get("derivation", ""),
                    teacher_model=self.model_name,
                    input_tokens=int(data.get("_input_tokens", 0) or 0),
                    output_tokens=int(data.get("_output_tokens", 0) or 0),
                )
            except (json.JSONDecodeError, asyncio.TimeoutError) as e:
                last_error = e
                logger.warning(
                    f"[distill {record_id}] attempt {attempt}/{self.max_retries} "
                    f"failed ({type(e).__name__}: {e}); retrying..."
                )
            except Exception as e:
                # Catch-all for safety: rate limit, 5xx, ...
                last_error = e
                msg = str(e).lower()
                if "rate" in msg or "quota" in msg or "429" in msg or "5" in msg[:3]:
                    logger.warning(
                        f"[distill {record_id}] attempt {attempt}/{self.max_retries} "
                        f"transient error: {e}; retrying..."
                    )
                else:
                    # Fatal error (e.g., auth) -> stop early.
                    logger.error(f"[distill {record_id}] fatal error: {e}")
                    raise TeacherClientError(str(e)) from e

            if attempt < self.max_retries:
                backoff = self.retry_backoff_s * (2 ** (attempt - 1))
                await asyncio.sleep(backoff)

        raise TeacherClientError(
            f"distill_one failed after {self.max_retries} attempts: {last_error}"
        )


def build_teacher_client() -> GeminiTeacherClient:
    """Factory returning client matching provider in setting.yaml."""
    from src.core.config import settings
    provider = settings.distillation.teacher.provider.lower()
    if provider == "gemini":
        return GeminiTeacherClient()
    raise ValueError(f"Unsupported teacher provider: {provider}")
