"""Physics RAG Node - truy xuat formula + example cho Coder model.

Goi 2 collection trong Qdrant (build boi scripts/rag/build_physics_index.py):
1. physics_formulas (per-topic formula sheet) -> top 1, lam fallback bao luon co cong thuc.
2. physics_examples (per-record problem + code) -> top 2, semantic match bai giong runtime question.

Format ket qua thanh 2 section ro rang trong context_block:
    RELEVANT FORMULAS:
    <topic + formulas>

    WORKED EXAMPLES:
    Example 1: ...
    Example 2: ...

Khi corpus chua build hoac retrieval that bai -> tra ve context="" (no-op fallback).
Formalizer se chay binh thuong khong co RAG.
"""
from src.agent.state import AgentState
from src.utils.logger import logger

# Ten collection - khop voi scripts/rag/build_physics_index.py.
PHYSICS_EXAMPLES_COLLECTION = "physics_examples"
PHYSICS_FORMULAS_COLLECTION = "physics_formulas"

# Tham so retrieval
TOP_K_EXAMPLES = 2
TOP_K_FORMULAS = 1
INITIAL_CANDIDATES = 12  # truoc rerank


def _format_context(formula_docs, example_docs) -> str:
    """Render 2 collection ket qua thanh 1 context_block 2 section."""
    parts: list[str] = []

    if formula_docs:
        parts.append("RELEVANT FORMULAS (apply these to derive the answer):")
        for d in formula_docs:
            content = d.node.get_content().strip()
            parts.append(content)
            parts.append("")  # blank line

    if example_docs:
        parts.append("WORKED EXAMPLES (reference the SymPy code style; do NOT blindly copy):")
        for i, d in enumerate(example_docs, start=1):
            content = d.node.get_content().strip()
            parts.append(f"Example {i}:")
            parts.append(content)
            parts.append("")

    return "\n".join(parts).strip()


def physics_rag_node(state: AgentState) -> dict:
    """Tim formula sheet + worked examples cho Coder model.

    Output:
        state["context"] = chuoi 2 section (co the rong neu retrieval fail).
    """
    try:
        from src.retrieval.engine import Retriever
        retriever = Retriever()
        question = state["question"]

        # Per-topic formulas (uu tien lay)
        formula_docs = []
        try:
            formula_docs = retriever.retrieval(
                query=question,
                collection_name=PHYSICS_FORMULAS_COLLECTION,
                k=INITIAL_CANDIDATES,
                mode="hybrid",
            )[:TOP_K_FORMULAS]
        except Exception as e:
            logger.warning(f"Physics RAG (formulas): {e}")

        # Per-record worked examples
        example_docs = []
        try:
            example_docs = retriever.retrieval(
                query=question,
                collection_name=PHYSICS_EXAMPLES_COLLECTION,
                k=INITIAL_CANDIDATES,
                mode="hybrid",
            )[:TOP_K_EXAMPLES]
        except Exception as e:
            logger.warning(f"Physics RAG (examples): {e}")

        context = _format_context(formula_docs, example_docs)
        if context:
            logger.info(
                f"Physics RAG: {len(formula_docs)} formula(s) + {len(example_docs)} example(s) "
                f"({len(context)} chars)."
            )
        else:
            logger.info("Physics RAG: khong tim duoc formula/example, formalizer chay khong RAG.")
    except Exception as e:
        logger.warning(f"Physics RAG that bai: {e}. Bo qua RAG.")
        context = ""

    return {"context": context}
