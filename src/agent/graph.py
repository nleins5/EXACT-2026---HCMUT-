"""
EXACT 2026 — Main LangGraph Pipeline
Graph flow:
    input → classify → {logic, physics}
    
    logic branch (Parallel):
        classify → logic_formalizer → logic_solver ─┐
        classify ─────────────────→ logic_direct ──┴─→ logic_explanation → END
    
    physics branch (Parallel):
        classify → physics_rag → physics_formalizer → physics_solver ─┐
        classify → physics_rag ─────────────────────→ physics_direct ─┴─→ physics_explanation → END
"""
from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes.classifier import classify_node, route_after_classify
from src.agent.nodes.logic_node import (
    logic_formalizer_node,
    logic_solver_node,
    logic_explanation_node,
    logic_direct_node,
)
from src.agent.nodes.physics_node import (
    physics_rag_node,
    physics_formalizer_node,
    physics_solver_node,
    physics_explanation_node,
    physics_direct_node,
)
from src.utils.logger import logger


def build_graph() -> StateGraph:
    """
    Xây dựng và biên dịch pipeline LangGraph hoàn chỉnh.
    
    Hàm này định nghĩa:
    1. Các Node (nút): Các đơn vị xử lý logic.
    2. Các Edge (cạnh): Luồng dữ liệu giữa các node.
    3. Conditional Edges: Rẽ nhánh dựa trên kết quả phân loại (logic vs physics).
    
    Returns:
        Compiled LangGraph workflow.
    """
    
    workflow = StateGraph(AgentState)

    # ── Đăng ký các Nodes ──────────────────────────────────────────────────
    workflow.add_node("classify",           classify_node)
    
    # Nhánh Logic (Dịch mã Z3 -> Giải mã -> Giải thích)
    workflow.add_node("logic_formalizer",   logic_formalizer_node)
    workflow.add_node("logic_solver",       logic_solver_node)
    workflow.add_node("logic_explanation",  logic_explanation_node)
    workflow.add_node("logic_direct",       logic_direct_node)
    
    # Nhánh Vật lý (RAG -> Dịch mã SymPy -> Giải mã -> Giải thích)
    workflow.add_node("physics_rag",        physics_rag_node)
    workflow.add_node("physics_formalizer", physics_formalizer_node)
    workflow.add_node("physics_solver",     physics_solver_node)
    workflow.add_node("physics_explanation", physics_explanation_node)
    workflow.add_node("physics_direct",     physics_direct_node)

    # ── Điểm bắt đầu (Entry Point) ──────────────────────────────────────────
    workflow.set_entry_point("classify")

    # ── Điều hướng điều kiện sau khi Phân loại ──────────────────────────────
    # Dựa vào task_type, graph sẽ fan-out (chạy song song) hoặc đi tiếp.
    # Dựa vào task_type, graph sẽ fan-out (chạy song song) hoặc đi tiếp.
    workflow.add_conditional_edges(
        "classify",
        route_after_classify,
        ["logic_formalizer", "logic_direct", "physics_rag"] # Danh sách các node đích có thể tới
    )

    # ── Luồng xử lý nhánh Logic ────────────────────────────────────────────
    workflow.add_edge("logic_formalizer",  "logic_solver")
    workflow.add_edge("logic_solver",      "logic_explanation")
    workflow.add_edge("logic_direct",      "logic_explanation") # Chờ cả 2 nhánh hội tụ
    workflow.add_edge("logic_explanation", END)

    # ── Luồng xử lý nhánh Vật lý ───────────────────────────────────────────
    workflow.add_edge("physics_rag",        "physics_formalizer")
    workflow.add_edge("physics_rag",        "physics_direct")     # Fan-out sau RAG
    workflow.add_edge("physics_formalizer", "physics_solver")
    workflow.add_edge("physics_solver",     "physics_explanation")
    workflow.add_edge("physics_direct",     "physics_explanation") # Chờ hội tụ
    workflow.add_edge("physics_explanation", END)

    return workflow.compile()


# Instance duy nhất của graph (Singleton pattern)
_graph = None

def get_graph():
    """Lấy hoặc khởi tạo instance của graph."""
    global _graph
    if _graph is None:
        logger.info("Đang biên dịch LangGraph pipeline...")
        _graph = build_graph()
    return _graph


def run_pipeline(
    question: str, 
    premises: list[str] = None,
    collection_name: str = "logic_regulations"
) -> dict:
    """
    Điểm đầu vào chính để chạy toàn bộ pipeline.
    
    Args:
        question: Câu hỏi từ người dùng.
        premises: Danh sách các giả thiết (nếu có - cho bài toán logic).
        collection_name: Tên collection trong Vector DB để truy vấn RAG.
        
    Returns:
        Một dictionary chứa đáp án, lập luận và mã code đã thực thi.
    """
    graph = get_graph()
    
    # Khởi tạo trạng thái ban đầu
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
            "confidence": 0.0
        },
        "fallback_answer": {
            "answer": "",
            "explanation": "",
            "fol": "",
            "cot": [],
            "premises": [],
            "confidence": 0.0
        },
        "error": "",
        "collection_name": collection_name,
        "context": "",
    }
    
    logger.info(f"Bắt đầu xử lý câu hỏi: {question[:100]}...")
    # Thực thi graph
    result = graph.invoke(initial_state)
    
    final = result.get("final_answer", {})
    intermediate = result.get("intermediate_answer", {})

    return {
        "task_type":    result.get("task_type"),
        "answer":       final.get("answer"),
        "explanation":  final.get("explanation"),
        "fol":          final.get("fol"),
        "cot":          final.get("cot"),
        "premises":     final.get("premises"),
        "confidence":   final.get("confidence"),
        "code":         intermediate.get("generated_code"),
        "code_output":  intermediate.get("code_output"),
    }
