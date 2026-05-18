from typing import Literal

from src.core.config import settings
from src.llm.provider.ollama_client import LlamaCppClient
from src.utils.logger import logger

# LlamaIndex Global Settings
from llama_index.core import Settings


class LLMFactory:
    """Factory + process-wide cache of :class:`LlamaCppClient` instances.

    Loading a 5GB GGUF on CPU costs ~30-60 seconds. The agent pipeline calls
    ``create_client`` once per node (formalizer, explanation, classifier...);
    without caching, each call builds a fresh client and reloads the same model
    file from disk. By keying the cache on ``(resolved_model_path, temperature)``
    we share a single :class:`ChatLlamaCpp` across nodes that use the same
    sampling settings, which removes the bulk of the per-request latency.
    """

    # Map purpose -> sampling temperature.
    # 0.0 for code/structured tasks (deterministic), higher for free-form summary.
    _PURPOSE_TEMPERATURE = {
        "rag":         0.0,
        "classifier":  0.0,
        "code":        0.0,
        "reasoning":   0.0,   # used by formalizer/classifier in agent nodes
        "summary":     0.0,
    }

    # Process-wide cache keyed by (model_path, temperature) so a single GGUF
    # is mmap'd into memory exactly once per (path, temp) combo.
    _client_cache: dict[tuple[str, float], LlamaCppClient] = {}

    @staticmethod
    def create_client(
        purpose: Literal["rag", "classifier", "code", "summary", "reasoning"],
        model_path: str = None,
    ) -> LlamaCppClient:
        import os
        use_mock = os.getenv("USE_MOCK_LLM", "false").lower() == "true"

        if use_mock:
            from src.llm.provider.mock_client import MockLLMClient
            logger.warning("USING MOCK LLM CLIENT (Offline Mode)")
            return MockLLMClient(model_path="mock", temperature=0.0)

        if model_path is None:
            model_path = settings.llm.model_path

        temperature = LLMFactory._PURPOSE_TEMPERATURE.get(purpose, 0.5)

        cache_key = (model_path, temperature)
        cached = LLMFactory._client_cache.get(cache_key)
        if cached is not None:
            logger.debug(
                f"Reusing cached LlamaCppClient for {purpose.upper()} "
                f"(path={model_path}, temp={temperature})"
            )
            return cached

        logger.debug(
            f"Creating LlamaCpp LLM for {purpose.upper()} "
            f"(model path: {model_path}, temp: {temperature})"
        )
        client = LlamaCppClient(model_path=model_path, temperature=temperature)
        LLMFactory._client_cache[cache_key] = client
        return client

    @staticmethod
    def clear_cache() -> None:
        """Drop all cached clients (useful for tests / hot-reload)."""
        LLMFactory._client_cache.clear()

    @staticmethod
    def configure_llama_index_settings(provider: str = "ollama"):
        logger.info(f"Configuring LlamaIndex global Settings for: {provider}")

        if provider == "llamacpp":
            from llama_index.llms.llama_cpp import LlamaCPP
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            Settings.llm = LlamaCPP(
                model_path=settings.llm.model_path,
                temperature=settings.llm.temperature,
            )
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=settings.embedding.model_name
            )

        logger.info(f"LlamaIndex global Settings updated successfully for {provider}")


def get_llm_provider(purpose: str = "rag", model_path: str = None):
    """Convenience helper to get a pre-configured LLM client instance."""
    return LLMFactory.create_client(purpose=purpose, model_path=model_path).get_llm()
