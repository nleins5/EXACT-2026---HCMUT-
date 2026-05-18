"""Physics RAG Node — truy xuat few-shot SymPy examples cho Coder model.

Vai tro: tim 2-3 bai vat ly tuong tu trong corpus (data/finetune/coder.jsonl
da loc Q19) -> dua vao prompt cua physics_formalizer nhu context_block, giup
Coder model sinh code SymPy chinh xac hon.

Format ket qua:
    Example 1:
    Problem: <problem text>
    Code:
    ```python
    <SymPy code>
    ```

    Example 2:
    ...

Khi corpus chua build hoac retrieval that bai -> tra ve context="" (no-op,
formalizer se chay binh thuong khong co RAG).
"""
from src.agent.state import AgentState
from src.utils.logger import logger

# Ten collection trong Qdrant — match scripts/rag/build_physics_index.py.
PHYSICS_COLLECTION = "physics_examples"


def _format_examples(docs) -> str:
    """Format LlamaIndex NodeWithScore list thanh few-shot block."""
    if not docs:
        return ""

    lines = []
    for i, d in enumerate(docs, start=1):
        content = d.node.get_content().strip()
        lines.append(f"Example {i}:\n{content}")
    return "\n\n".join(lines)


def physics_rag_node(state: AgentState) -> dict:
    """Tim few-shot examples SymPy lien quan tu Vector DB.

    Dau ra: state["context"] = chuoi few-shot block (co the rong).
    """
    try:
        from src.retrieval.engine import Retriever
        retriever = Retriever()
        docs = retriever.retrieval(
            query=state["question"],
            collection_name=PHYSICS_COLLECTION,
            mode="hybrid",
        )
        context = _format_examples(docs)
        if context:
            logger.info(f"Physics RAG: {len(docs)} few-shot examples ({len(context)} chars).")
        else:
            logger.info("Physics RAG: khong tim duoc example, formalizer chay khong co RAG.")
    except Exception as e:
        logger.warning(f"Physics RAG that bai: {e}. Bo qua RAG.")
        context = ""

    return {"context": context}
