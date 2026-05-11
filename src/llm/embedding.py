import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parents[2]))

from src.core.config import settings
from src.utils.logger import logger
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class EmbeddingFactory:
    def __init__(self):
        self.embedding = settings.embedding.model_name

    def get_embedding(self) -> HuggingFaceEmbedding:
        if self.embedding is None:
            self.embedding = "BAAI/bge-m3"

        logger.info(f"Loading HF embedding model: {self.embedding}")

        return HuggingFaceEmbedding(
            model_name=self.embedding,
        )
