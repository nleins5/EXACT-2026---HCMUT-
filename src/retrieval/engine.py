"""Retrieval engine — hybrid (vector + BM25) + reranker.

Wrapper mong tren LlamaIndex Hybrid Retriever. Su dung trong physics_rag_node
de cap context cho physics_formalizer (sinh code SymPy chinh xac hon).
"""
from llama_index.core.postprocessor import SentenceTransformerRerank

from src.agent.llm.embedding import EmbeddingFactory
from src.core.config import settings
from src.retrieval.vector_db import VectorDBManager
from src.utils.logger import logger


class Retriever:
    """Hybrid search engine (vector + BM25) co reranker."""

    def __init__(self):
        embed_factory = EmbeddingFactory()
        self.embedding_model = embed_factory.get_embedding()

        self.vector_db = VectorDBManager(embedding_model=self.embedding_model)

        # Reranker — bao toi 3 doc relevant nhat de tiet kiem token.
        rerank_model = settings.rag.reranker
        logger.info(f"Loading reranker model: {rerank_model}")
        self.reranker = SentenceTransformerRerank(
            model=rerank_model,
            top_n=3,
        )

        logger.info("Retriever initialized successfully.")

    def retrieval(
        self,
        query: str,
        collection_name: str = "default_collection",
        k: int = 20,            # so candidate ban dau truoc rerank
        mode: str = "hybrid",
    ):
        """Retrieve documents bang hybrid search + rerank top 3.

        Args:
            query: chuoi query.
            collection_name: ten Qdrant collection.
            k: top-k truoc rerank.
            mode: "hybrid" (vector + BM25) hoac "vector".

        Returns:
            List[NodeWithScore] (toi da 3 doc).
        """
        if mode == "hybrid":
            logger.info(f"[Hybrid Search] Top {k} candidates for: {query}")
            retriever = self.vector_db.get_hybrid_retriever(
                similarity_top_k=k,
                collection_name=collection_name,
            )
        else:
            logger.info(f"[Vector Search] Top {k} candidates for: {query}")
            retriever = self.vector_db.get_retriever(
                similarity_top_k=k,
                collection_name=collection_name,
            )

        if retriever is None:
            logger.warning(f"⚠️ No index found for collection '{collection_name}'.")
            return []

        try:
            initial_nodes = retriever.retrieve(query)
            logger.info(f"Retrieved {len(initial_nodes)} initial candidates.")

            if not initial_nodes:
                return []

            logger.info(f"Reranking to top 3 using {settings.rag.reranker}...")
            reranked_nodes = self.reranker.postprocess_nodes(
                initial_nodes,
                query_str=query,
            )
            logger.info(f"Reranked: {len(reranked_nodes)} documents.")
            return reranked_nodes

        except Exception as e:
            logger.error(f"Loi retrieval/reranking: {e}")
            return []
