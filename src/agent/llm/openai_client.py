"""OpenAI-compatible client targeting local llama-server."""
from __future__ import annotations

from typing import Any, Type

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.agent.llm.base import BaseLLM
from src.core.config import settings
from src.utils.logger import logger


class OpenAILLMClient(BaseLLM):
    """Wraps ChatOpenAI pointed at local llama-server."""

    def __init__(self, role: str = "instruct"):
        cfg_server = settings.llm.server
        cfg_role = (
            settings.llm.coder if role == "coder"
            else settings.llm.instruct
        )

        self.role = role
        self.model_name = cfg_role.model_name
        self.temperature = cfg_role.temperature
        self.max_tokens = cfg_role.max_tokens
        self.base_url = cfg_server.base_url
        self.api_key = cfg_server.api_key

        self._llm: ChatOpenAI | None = None

        logger.debug(
            f"OpenAILLMClient init (role={role}, model={self.model_name}, "
            f"base_url={self.base_url})"
        )

    def get_llm(self, **kwargs) -> ChatOpenAI:
        """Return a singleton ChatOpenAI bound to the current role."""
        if self._llm is not None:
            return self._llm

        self._llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            base_url=self.base_url,
            api_key=self.api_key,
            max_retries=0,
            timeout=float(settings.api.request_budget_seconds),
            **kwargs,
        )
        return self._llm

    def get_structured_llm(self, output_schema: Type[BaseModel]) -> Any:
        """Return LLM with Pydantic structured output enforcement."""
        llm = self.get_llm()
        return llm.with_structured_output(output_schema)
