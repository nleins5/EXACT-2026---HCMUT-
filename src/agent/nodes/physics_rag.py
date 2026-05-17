"""
Physics RAG Node — Truy xuất kiến thức vật lý từ Vector DB.
"""
from src.agent.state import AgentState
from src.utils.logger import logger


def physics_rag_node(state: AgentState) -> dict:
    """Tìm kiếm công thức vật lý liên quan từ cơ sở tri thức."""
    try:
        from src.retrieval.engine import Retriever
        retriever = Retriever()
        docs = retriever.retrieval(
            query=state["question"],
            collection_name="physics_knowledge",
            mode="hybrid",
        )
        if docs:
            context = "\n\n".join([d.node.get_content() for d in docs])
            logger.info(f"Physics RAG: {len(docs)} tài liệu liên quan.")
        else:
            context = "Không tìm thấy công thức cụ thể trong cơ sở tri thức."
            logger.info("Physics RAG: Không tìm thấy tài liệu nào.")
    except Exception as e:
        logger.warning(f"Physics RAG thất bại: {e}")
        context = ""

    return {"context": context}
