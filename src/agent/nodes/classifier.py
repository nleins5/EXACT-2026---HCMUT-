"""Classifier Node — phan loai cau hoi thanh logic / physics.

Logic phan loai (rule-based, KHONG goi LLM):
- Neu evaluator gui explicit `task_type` / `query_type` -> ton trong field do.
- Co `premises[]` -> logic (Type 1, BTC slide 12).
- Khong co premises -> physics (Type 2).

Lua chon nay tiet kiem 1 lan swap LLM/request — phu hop budget 60s.
"""
from src.agent.state import AgentState
from src.utils.logger import logger


def classify_node(state: AgentState) -> dict:
    """Phan loai cau hoi thanh logic hoac physics dua tren su ton tai cua premises."""
    requested_task_type = state.get("requested_task_type")
    if requested_task_type in {"logic", "physics"}:
        logger.info("Classifier: explicit task_type=%s from request.", requested_task_type)
        return {"task_type": requested_task_type}

    premises = state.get("premises", []) or []

    if premises:
        task_type = "logic"
        logger.info(f"Classifier: phat hien {len(premises)} premises -> logic (Type 1).")
    else:
        task_type = "physics"
        logger.info("Classifier: khong co premises -> physics (Type 2).")

    return {"task_type": task_type}


def route_after_classify(state: AgentState) -> str:
    """Router — quyet dinh node ke tiep dua tren task_type.

    Tra ve ten 1 node duy nhat (sequential, khong fan-out).
    """
    t_type = state.get("task_type", "logic")
    if t_type == "physics":
        return "physics_rag"
    return "logic_formalizer"
