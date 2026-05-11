from typing import Any, Type

from langchain_community.chat_models import ChatLlamaCpp
from pydantic import BaseModel

from src.core.config import settings
from src.llm.base import BaseLLM
from src.utils.logger import logger


class LlamaCppClient(BaseLLM):
    """Client for locally hosted GGUF models using LlamaCpp directly (Serverless).

    Exposes both plain and structured-output LLM variants compatible with LangChain.
    """

    def __init__(self, model_path: str, temperature: float = 0.0):
        """Initialise the client.

        Args:
            model_path:  Path to the GGUF model file locally.
            temperature: Sampling temperature; 0.0 for deterministic output.
        """
        self.model_path = model_path
        self.temperature = temperature

        logger.debug(
            f"OllamaClient (LlamaCpp) initialized — model_path: {self.model_path}, "
            f"temperature: {self.temperature}"
        )

    def get_llm(self, **kwargs) -> ChatLlamaCpp:
        """Build and return a ``LlamaCpp`` instance.

        Args:
            **kwargs: Additional keyword arguments forwarded to ``LlamaCpp``.

        Returns:
            A configured ``LlamaCpp`` model ready for invocation.
        """
        logger.debug(f"Building ChatLlamaCpp instance (path: {self.model_path})")

        if self.model_path is None:
            self.model_path = settings.llm.model_path

        if self.temperature is None:
            self.temperature = 0.0

        return ChatLlamaCpp(
            model_path=self.model_path,
            temperature=self.temperature,
            **kwargs,
        )

    def get_structured_llm(self, output_schema: Type[BaseModel]) -> Any:
        """Return an LLM bound to a Pydantic schema for structured output.

        Args:
            output_schema: Pydantic model class the LLM response is parsed into.

        Returns:
            A model instance that enforces the given output schema.
        """
        logger.debug(f"Building structured LLM for schema: {output_schema.__name__}")
        model = self.get_llm()
        # Đối với LlamaCpp, việc dùng .with_structured_output phụ thuộc vào phiên bản langchain-community
        return model.with_structured_output(output_schema)
