"""LLMFactory — single-resident gateway to llama-server."""
from __future__ import annotations

from typing import Literal, Optional

from src.agent.llm.openai_client import OpenAILLMClient
from src.agent.llm.server_supervisor import LlamaServerSupervisor
from src.utils.logger import logger

Role = Literal["coder", "instruct"]


class LLMFactory:

    _supervisor: Optional[LlamaServerSupervisor] = None
    _client: Optional[OpenAILLMClient] = None
    _active_role: Optional[Role] = None

    @classmethod
    def init(cls, supervisor: LlamaServerSupervisor) -> None:
        cls._supervisor = supervisor
        cls._client = None
        cls._active_role = None
        logger.info("LLMFactory initialised.")

    @classmethod
    def activate(cls, role: Role) -> OpenAILLMClient:
        if cls._supervisor is None:
            raise RuntimeError("LLMFactory not initialised. Call LLMFactory.init(supervisor) in app lifespan.")

        if cls._active_role != role:
            cls._supervisor.swap_to(role)
            cls._active_role = role
            cls._client = OpenAILLMClient(role=role)
            logger.info(
                f"LLMFactory.activate({role}) — process swapped, client rebuilt."
            )
        elif cls._client is None:
            cls._client = OpenAILLMClient(role=role)

        return cls._client

    @classmethod
    def reset(cls) -> None:
        cls._supervisor = None
        cls._client = None
        cls._active_role = None
