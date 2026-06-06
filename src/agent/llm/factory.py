"""LLMFactory — single-resident gateway to llama-server."""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Literal, Optional

from src.agent.llm.openai_client import OpenAILLMClient
from src.agent.llm.server_supervisor import LlamaServerSupervisor
from src.utils.logger import logger

Role = Literal["coder", "instruct"]


class LLMFactory:

    _supervisor: Optional[LlamaServerSupervisor] = None
    _client: Optional[OpenAILLMClient] = None
    _active_role: Optional[Role] = None
    _generation: int = 0
    _state_lock = threading.RLock()
    _activation_lock = threading.Lock()
    _pipeline_lock = threading.Lock()

    @classmethod
    def init(cls, supervisor: LlamaServerSupervisor) -> None:
        with cls._state_lock:
            cls._supervisor = supervisor
            cls._client = None
            cls._active_role = None
            cls._generation += 1
        logger.info("LLMFactory initialised.")

    @classmethod
    def activate(cls, role: Role) -> OpenAILLMClient:
        with cls._activation_lock:
            with cls._state_lock:
                supervisor = cls._supervisor
                generation = cls._generation
                if supervisor is None:
                    raise RuntimeError(
                        "LLMFactory not initialised. Call LLMFactory.init(supervisor) in app lifespan."
                    )
                if cls._active_role == role and cls._client is not None and supervisor.is_alive():
                    return cls._client

            supervisor.swap_to(role)

            with cls._state_lock:
                if generation != cls._generation or cls._supervisor is not supervisor:
                    raise RuntimeError("LLM activation was cancelled.")
                cls._active_role = role
                cls._client = OpenAILLMClient(role=role)
                logger.info(
                    f"LLMFactory.activate({role}) — process swapped, client rebuilt."
                )
                return cls._client

    @classmethod
    @contextmanager
    def pipeline_session(cls, cancel_event: threading.Event | None = None):
        """Serialize complete pipelines so model swaps cannot corrupt another request."""
        while not cls._pipeline_lock.acquire(timeout=0.1):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Pipeline cancelled while waiting for the request gate.")
        try:
            yield
        finally:
            cls._pipeline_lock.release()

    @classmethod
    def abort_active_request(cls) -> None:
        """Interrupt an in-flight llama-server call and invalidate its client."""
        with cls._state_lock:
            supervisor = cls._supervisor
            cls._generation += 1
            cls._active_role = None
            cls._client = None
        if supervisor is not None:
            supervisor.shutdown()

    @classmethod
    def is_ready(cls) -> bool:
        with cls._state_lock:
            return bool(cls._supervisor and cls._supervisor.is_alive() and cls._active_role)

    @classmethod
    def is_busy(cls) -> bool:
        return cls._pipeline_lock.locked()

    @classmethod
    def reset(cls) -> None:
        with cls._state_lock:
            cls._supervisor = None
            cls._client = None
            cls._active_role = None
            cls._generation += 1
