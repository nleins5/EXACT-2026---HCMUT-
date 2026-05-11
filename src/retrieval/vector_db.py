from src.core.config import settings
from src.utils.logger import logger
import json
from pathlib import Path
import shutil
import os
import gc

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.retrievers import AutoMergingRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever


class VectorDBManager:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

        # Set up base storage path
        self.base_storage_dir = Path(__file__).resolve().parents[2] / "storage"
        self.base_storage_dir.mkdir(parents=True, exist_ok=True)

        # Separate sub-directory for Qdrant segment files
        self._qdrant_path = self.base_storage_dir / "qdrant_storage"
        self._qdrant_path.mkdir(parents=True, exist_ok=True)

        self._db_client = None
        self._index = None

    @property
    def db_client(self) -> QdrantClient:
        """Lazy-initialise the local (embedded) QdrantClient."""
        if self._db_client is None:
            self._db_client = QdrantClient(path=str(self._qdrant_path))
        return self._db_client

    def _get_embedding_dimension(self) -> int:
        """Detect the vector dimension of the current embedding model."""
        try:
            sample = self.embedding_model.get_text_embedding("dimension test")
            return len(sample)
        except Exception as e:
            logger.warning(f"⚠️ Cannot detect embed_dim, defaulting to 1024 (BGE-M3): {e}")
            return 1024

    def _collection_exists(self, collection_name: str) -> bool:
        existing = [c.name for c in self.db_client.get_collections().collections]
        return collection_name in existing

    def get_persist_dir(self, collection_name: str) -> Path:
        pdir = self.base_storage_dir / collection_name
        pdir.mkdir(parents=True, exist_ok=True)
        return pdir

    def _get_storage_context(self, collection_name: str, persist_dir: Path = None) -> StorageContext:
        if not self._collection_exists(collection_name):
            embed_dim = self._get_embedding_dimension()
            self.db_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
            )
            logger.info(f"✅ Created Qdrant collection '{collection_name}' (dim={embed_dim})")

        vector_store = QdrantVectorStore(
            client=self.db_client,
            collection_name=collection_name,
        )

        if persist_dir is None:
            persist_dir = self.get_persist_dir(collection_name)

        return StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=str(persist_dir) if (persist_dir / "docstore.json").exists() else None
        )

    def reset_db(self):
        """Completely wipe all Qdrant data and LlamaIndex metadata."""
        try:
            if self._db_client is not None:
                self._db_client.close()
            self._index = None
            self._db_client = None
            gc.collect()

            if self.base_storage_dir.exists():
                shutil.rmtree(self.base_storage_dir, ignore_errors=True)
                logger.info(f"🧹 Cleaned up storage directory: {self.base_storage_dir}")

            self.base_storage_dir.mkdir(parents=True, exist_ok=True)
            self._qdrant_path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"❌ Failed to reset DB: {e}")
            return False

    def get_index(self, collection_name: str = "default_collection"):
        if self._index is not None:
            return self._index

        try:
            persist_dir = self.get_persist_dir(collection_name)
            if (persist_dir / "docstore.json").exists():
                storage_context = self._get_storage_context(collection_name, persist_dir=persist_dir)
                self._index = load_index_from_storage(
                    storage_context,
                    embed_model=self.embedding_model,
                )
                return self._index
            return None
        except Exception as e:
            logger.warning(f"⚠️ Error loading index for '{collection_name}': {e}")
            return None

    def add_documents(self, documents, collection_name: str = "default_collection"):
        """Insert documents into the index and persist."""
        try:
            self._index = self.get_index(collection_name)
            if self._index is None:
                storage_context = self._get_storage_context(collection_name)
                self._index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=storage_context,
                    embed_model=self.embedding_model,
                    show_progress=True,
                )
            else:
                for doc in documents:
                    self._index.insert(doc)

            persist_dir = self.get_persist_dir(collection_name)
            self._index.storage_context.persist(persist_dir=str(persist_dir))
            logger.info(f"✅ Persisted index for '{collection_name}'")
        except Exception as e:
            logger.error(f"❌ Error in add_documents: {e}")
            raise

    def get_retriever(self, similarity_top_k: int = 20, collection_name: str = "default_collection"):
        index = self.get_index(collection_name)
        if index is None:
            return None
        return index.as_retriever(similarity_top_k=similarity_top_k)

    def get_hybrid_retriever(self, similarity_top_k: int = 20, collection_name: str = "default_collection"):
        index = self.get_index(collection_name)
        if index is None:
            return None

        all_nodes = list(index.storage_context.docstore.docs.values())
        if not all_nodes:
            return self.get_retriever(similarity_top_k=similarity_top_k, collection_name=collection_name)

        vector_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=all_nodes,
            similarity_top_k=similarity_top_k,
        )

        return QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            similarity_top_k=similarity_top_k,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=False,
            verbose=True,
        )
