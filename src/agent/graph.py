"""
EXACT 2026 — Main LangGraph Pipeline
Graph flow:
    input -> classify -> {logic, physics}
    
    logic branch (Parallel):
        classify -> logic_formalizer -> logic_solver -+
        classify -> logic_direct ----+-> logic_explanation -> END
    
    physics branch (Parallel):
        classify -> physics_rag -> physics_formalizer -> physics_solver -+
        classify -> physics_rag -> physics_direct ------+-> physics_explanation -> END
"""
from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes.classifier import classify_node, route_after_classify
from src.agent.nodes.logic_formalizer import logic_formalizer_node
from src.agent.nodes.logic_solver import logic_solver_node
from src.agent.nodes.logic_explanation import logic_explanation_node
from src.agent.nodes.logic_direct import logic_direct_node
from src.agent.nodes.physics_rag import physics_rag_node
from src.agent.nodes.physics_formalizer import physics_formalizer_node
from src.agent.nodes.physics_solver import physics_solver_node
from src.agent.nodes.physics_explanation import physics_explanation_node
from src.agent.nodes.physics_direct import physics_direct_node
from src.utils.logger import logger


def build_graph() -> StateGraph:
    """Xây dựng và biên dịch pipeline LangGraph hoàn chỉnh."""

    workflow = StateGraph(AgentState)

    # ── Nodes ──
    workflow.add_node("classify",            classify_node)

    workflow.add_node("logic_formalizer",     logic_formalizer_node)
    workflow.add_node("logic_solver",         logic_solver_node)
    workflow.add_node("logic_explanation",    logic_explanation_node)
    workflow.add_node("logic_direct",         logic_direct_node)

    workflow.add_node("physics_rag",          physics_rag_node)
    workflow.add_node("physics_formalizer",   physics_formalizer_node)
    workflow.add_node("physics_solver",       physics_solver_node)
    workflow.add_node("physics_explanation",  physics_explanation_node)
    workflow.add_node("physics_direct",       physics_direct_node)

    # ── Entry Point ──
    workflow.set_entry_point("classify")

    # ── Conditional Routing ──
    workflow.add_conditional_edges(
        "classify",
        route_after_classify,
        ["logic_formalizer", "logic_direct", "physics_rag"],
    )

    # ── Logic Branch ──
    workflow.add_edge("logic_formalizer",  "logic_solver")
    workflow.add_edge("logic_solver",      "logic_explanation")
    workflow.add_edge("logic_direct",      "logic_explanation")
    workflow.add_edge("logic_explanation", END)

    # ── Physics Branch ──
    workflow.add_edge("physics_rag",        "physics_formalizer")
    workflow.add_edge("physics_rag",        "physics_direct")
    workflow.add_edge("physics_formalizer", "physics_solver")
    workflow.add_edge("physics_solver",     "physics_explanation")
    workflow.add_edge("physics_direct",     "physics_explanation")
    workflow.add_edge("physics_explanation", END)

    return workflow.compile()


# Singleton
_graph = None


def get_graph():
    """Lấy hoặc khởi tạo instance của graph."""
    global _graph
    if _graph is None:
        logger.info("Compiling LangGraph pipeline...")
        _graph = build_graph()
    return _graph


def run_pipeline(
    question: str,
    premises: list[str] = None,
    collection_name: str = "logic_regulations",
) -> dict:
    """
    Điểm đầu vào chính để chạy toàn bộ pipeline.

    Args:
        question: Câu hỏi từ người dùng.
        premises: Danh sách giả thiết (cho bài toán logic).
        collection_name: Tên collection Vector DB cho RAG.

    Returns:
        Dictionary chứa đáp án, lập luận và code đã thực thi.
    """
    graph = get_graph()

    initial_state: AgentState = {
        "question": question,
        "premises": premises or [],
        "task_type": "logic",
        "intermediate_answer": {
            "generated_code": "",
            "code_output": "",
        },
        "final_answer": {
            "answer": "",
            "explanation": "",
            "fol": "",
            "cot": [],
            "premises": [],
            "confidence": 0.0,
        },
        "fallback_answer": {
            "answer": "",
            "explanation": "",
            "fol": "",
            "cot": [],
            "premises": [],
            "confidence": 0.0,
        },
        "error": "",
        "collection_name": collection_name,
        "context": "",
    }

    logger.info(f"Processing: {question[:100]}...")
    result = graph.invoke(initial_state)

    final = result.get("final_answer", {})
    intermediate = result.get("intermediate_answer", {})

    return {
        "task_type":   result.get("task_type"),
        "answer":      final.get("answer"),
        "explanation": final.get("explanation"),
        "fol":         final.get("fol"),
        "cot":         final.get("cot"),
        "premises":    final.get("premises"),
        "confidence":  final.get("confidence"),
        "code":        intermediate.get("generated_code"),
        "code_output": intermediate.get("code_output"),
    }
