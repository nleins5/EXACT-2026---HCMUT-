"""Embedding factory — dung cho RAG (LlamaIndex hybrid retriever)."""
import sys
from pathlib import Path

# Them project root vao sys.path de import src.* khi chay standalone.
sys.path.append(str(Path(__file__).parents[3]))

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.core.config import settings
from src.utils.logger import logger


class EmbeddingFactory:
    """Tra ve `HuggingFaceEmbedding` da configure tu setting.yaml.

    Mac dinh model: `BAAI/bge-m3` — multilingual, hop voi tieng Viet + cong thuc.
    """

    def __init__(self):
        self.embedding = settings.embedding.model_name

    def get_embedding(self) -> HuggingFaceEmbedding:
        if self.embedding is None:
            self.embedding = "BAAI/bge-m3"

        logger.info(f"Loading HF embedding model: {self.embedding}")

        return HuggingFaceEmbedding(
            model_name=self.embedding,
        )
