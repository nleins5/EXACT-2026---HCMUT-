from typing import Literal

from src.core.config import settings
from src.llm.provider.ollama_client import LlamaCppClient
from src.utils.logger import logger

# LlamaIndex Global Settings
from llama_index.core import Settings


class LLMFactory:

    @staticmethod
    def create_client(
        purpose: Literal["rag", "classifier", "code", "summary"], 
        model_path: str = None
    ) -> LlamaCppClient:

        import os
        use_mock = os.getenv("USE_MOCK_LLM", "false").lower() == "true"

        if use_mock:
            from src.llm.provider.mock_client import MockLLMClient
            logger.warning("USING MOCK LLM CLIENT (Offline Mode)")
            return MockLLMClient(model_path="mock", temperature=0.0)

        if model_path is None:
            model_path = settings.llm.model_path

        if purpose in ("rag", "classifier"):
            temperature = 0.0
            logger.debug(
                f"Creating Ollama LLM for {purpose.upper()} (model path: {model_path}, temp: {temperature})"
            )
            return LlamaCppClient(model_path=model_path, temperature=temperature)
        elif purpose == "code":
            temperature = 0.0
            logger.debug(
                f"Creating Ollama LLM for {purpose.upper()} (model path: {model_path}, temp: {temperature})"
            )
            return LlamaCppClient(model_path=model_path, temperature=temperature)
        elif purpose == "summary":
            temperature = 0.0
            logger.debug(
                f"Creating Ollama LLM for {purpose.upper()} (model path: {model_path}, temp: {temperature})"
            )
            return LlamaCppClient(model_path=model_path, temperature=temperature)
        else:
            temperature = 0.5
            logger.debug(
                f"Creating Ollama LLM for {purpose.upper()} (model path: {model_path}, temp: {temperature})"
            )
            return LlamaCppClient(model_path=model_path, temperature=temperature) # Fix: using LlamaCppClient instead of non-existent OllamaClient

    @staticmethod
    def configure_llama_index_settings(provider: str = "ollama"):

        logger.info(f"Configuring LlamaIndex global Settings for: {provider}")

        if provider == "llamacpp":
            from llama_index.llms.llama_cpp import LlamaCPP
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            # Sử dụng LlamaCPP wrapper của LlamaIndex để đồng bộ với LangChain
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
