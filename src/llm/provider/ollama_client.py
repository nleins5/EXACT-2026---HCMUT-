from typing import Any, Type
from pathlib import Path

from langchain_community.chat_models import ChatLlamaCpp
from pydantic import BaseModel

from src.core.config import settings
from src.llm.base import BaseLLM
from src.utils.logger import logger


class LlamaCppClient(BaseLLM):
    """Client for locally hosted GGUF models using LlamaCpp directly (Serverless).

    Exposes both plain and structured-output LLM variants compatible with LangChain.

    The underlying ``ChatLlamaCpp`` is loaded **lazily** on first ``get_llm()``
    call and cached on the instance. Reuse this client (via
    :class:`src.llm.factory.LLMFactory`) across multiple agent nodes instead of
    instantiating a new one per node — loading a 5GB GGUF on CPU costs 30-60s.
    """

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.0,
        n_ctx: int | None = None,
        n_gpu_layers: int | None = None,
    ):
        """Initialise the client.

        Args:
            model_path:    Path to the GGUF model file locally.
            temperature:   Sampling temperature; 0.0 for deterministic output.
            n_ctx:         Context window size. None = read from settings.
            n_gpu_layers:  Number of layers to offload to GPU. None = read from settings.
                           0 = CPU only, -1 = all layers on GPU.
        """
        # Resolve model_path: if relative, anchor to project root
        if model_path:
            mp = Path(model_path)
            if not mp.is_absolute():
                project_root = Path(__file__).resolve().parents[3]
                mp = project_root / mp
            self.model_path = str(mp)
        else:
            self.model_path = settings.llm.model_path

        self.temperature = temperature
        self.n_ctx = n_ctx if n_ctx is not None else settings.llm.n_ctx
        self.n_gpu_layers = (
            n_gpu_layers if n_gpu_layers is not None else settings.llm.n_gpu_layers
        )

        # Lazy-loaded instance — only built on first get_llm() call.
        self._llm: ChatLlamaCpp | None = None

        logger.debug(
            f"LlamaCppClient initialized - model_path: {self.model_path}, "
            f"temperature: {self.temperature}, n_ctx: {self.n_ctx}, "
            f"n_gpu_layers: {self.n_gpu_layers}"
        )

    def get_llm(self, **kwargs) -> ChatLlamaCpp:
        """Build (once) and return a cached ``ChatLlamaCpp`` instance.

        Args:
            **kwargs: Additional keyword arguments forwarded to ``ChatLlamaCpp``
                on the FIRST call. Subsequent calls reuse the cached instance and
                ignore overrides — instantiate a new client if you need different
                parameters.

        Returns:
            A configured ``ChatLlamaCpp`` model ready for invocation.
        """
        if self._llm is not None:
            logger.debug("Reusing cached ChatLlamaCpp instance.")
            return self._llm

        logger.info(f"Loading ChatLlamaCpp from {self.model_path} (this may take a while)")

        default_kwargs = {
            "model_path":    self.model_path,
            "temperature":   self.temperature,
            "n_ctx":         self.n_ctx,
            "n_gpu_layers":  self.n_gpu_layers,
            "verbose":       False,
            # Bumped from 1024 -> 2048 so DeepSeek-R1's <think> block plus the
            # actual code fence usually fit; otherwise output is truncated and
            # the formalizer ends up emitting an open fence we then have to
            # salvage via the error-prompt path.
            "max_tokens":    2048,
        }
        default_kwargs.update(kwargs)

        self._llm = ChatLlamaCpp(**default_kwargs)
        return self._llm

    def get_structured_llm(self, output_schema: Type[BaseModel]) -> Any:
        """Return an LLM bound to a Pydantic schema for structured output.

        Args:
            output_schema: Pydantic model class the LLM response is parsed into.

        Returns:
            A model instance that enforces the given output schema.
        """
        logger.debug(f"Building structured LLM for schema: {output_schema.__name__}")
        model = self.get_llm()
        return model.with_structured_output(output_schema)
