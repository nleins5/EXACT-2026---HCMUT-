from src.core.config import settings
from src.utils.logger import logger
from src.retrieval.vector_db import VectorDBManager
from src.llm.embedding import EmbeddingFactory
from llama_index.core.postprocessor import SentenceTransformerRerank


class Retriever():
    def __init__(self):
        embed_factory = EmbeddingFactory()
        self.embedding_model = embed_factory.get_embedding()

        self.vector_db = VectorDBManager(embedding_model=self.embedding_model)
        
        # Initialise Reranker
        rerank_model = settings.rag.reranker
        logger.info(f"Loading reranker model: {rerank_model}")
        self.reranker = SentenceTransformerRerank(
            model=rerank_model, 
            top_n=3 # As requested: "re rank lại với 3 index thôi"
        )
    
        logger.info("Retriever initialized successfully.")

    def retrieval(
        self,
        query: str,
        collection_name: str = "default_collection",
        k: int = 20, # Initial retrieval top k
        mode: str = "hybrid",
    ):
        """Retrieve documents using hybrid search and then rerank."""
        
        # 1. Get appropriate retriever
        if mode == "hybrid":
            logger.info(f"[Hybrid Search] Retrieving top {k} candidates for: {query}")
            retriever = self.vector_db.get_hybrid_retriever(
                similarity_top_k=k,
                collection_name=collection_name
            )
        else:
            logger.info(f"[Vector Search] Retrieving top {k} candidates for: {query}")
            retriever = self.vector_db.get_retriever(
                similarity_top_k=k,
                collection_name=collection_name
            )

        if retriever is None:
            logger.warning(f"⚠️ No index found for collection '{collection_name}'.")
            return []

        try:
            # 2. Initial retrieval
            initial_nodes = retriever.retrieve(query)
            logger.info(f"Retrieved {len(initial_nodes)} initial candidates.")

            if not initial_nodes:
                return []

            # 3. Reranking (Top 3)
            logger.info(f"Reranking candidates to top 3 using {settings.rag.reranker}...")
            reranked_nodes = self.reranker.postprocess_nodes(
                initial_nodes, 
                query_str=query
            )
            
            logger.info(f"Successfully reranked to {len(reranked_nodes)} documents.")
            return reranked_nodes

        except Exception as e:
            logger.error(f"Error during retrieval/reranking: {e}")
            return []
