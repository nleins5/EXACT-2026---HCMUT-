"""EXACT 2026 — LangGraph Pipeline (Sequential + Retry Loop)."""
from typing import Literal

from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes.classifier import classify_node, route_after_classify
from src.agent.nodes.logic_formalizer import logic_formalizer_node
from src.agent.nodes.logic_solver import logic_solver_node
from src.agent.nodes.logic_explanation import logic_explanation_node
from src.agent.nodes.logic_direct import logic_direct_node, should_use_logic_direct
from src.agent.nodes.logic_retrieval import retrieve_known_logic
from src.agent.nodes.physics_rag import physics_rag_node
from src.agent.nodes.physics_formalizer import physics_formalizer_node
from src.agent.nodes.physics_solver import physics_solver_node
from src.agent.nodes.physics_explanation import physics_explanation_node
from src.agent.nodes.physics_baseline import solve_common_physics
from src.core.config import settings
from src.agent.runtime import cancellation_context, cancellation_guard
from src.utils.logger import logger

_MAX_RETRIES = settings.solver.max_retries


# ── Retry routing functions ──────────────────────────────────────────


def _route_after_logic_solver(state: AgentState) -> str:
    intermediate = state.get("intermediate_answer", {})
    retry_count = state.get("retry_count", 0)

    if intermediate.get("code_error", False) and retry_count < _MAX_RETRIES:
        logger.info(
            f"Logic solver failed (retry {retry_count + 1}/{_MAX_RETRIES}), "
            f"retrying formalizer with error feedback."
        )
        return "logic_formalizer_retry"
    return "logic_explanation"


def _route_after_physics_solver(state: AgentState) -> str:
    intermediate = state.get("intermediate_answer", {})
    retry_count = state.get("retry_count", 0)

    if intermediate.get("code_error", False) and retry_count < _MAX_RETRIES:
        logger.info(
            f"Physics solver failed (retry {retry_count + 1}/{_MAX_RETRIES}), "
            f"retrying formalizer with error feedback."
        )
        return "physics_formalizer_retry"
    return "physics_explanation"





def _logic_retry_node(state: AgentState) -> dict:
    retry_count = state.get("retry_count", 0) + 1
    intermediate = state.get("intermediate_answer", {})
    intermediate["retry_error_feedback"] = intermediate.get("error_message", "")
    logger.info(f"Logic retry node: attempt {retry_count}")
    result = logic_formalizer_node(state)
    result["retry_count"] = retry_count
    return result


def _physics_retry_node(state: AgentState) -> dict:
    retry_count = state.get("retry_count", 0) + 1
    intermediate = state.get("intermediate_answer", {})
    intermediate["retry_error_feedback"] = intermediate.get("error_message", "")
    logger.info(f"Physics retry node: attempt {retry_count}")
    result = physics_formalizer_node(state)
    result["retry_count"] = retry_count
    return result


def build_graph() -> StateGraph:

    workflow = StateGraph(AgentState)


    workflow.add_node("classify",             cancellation_guard(classify_node))

    workflow.add_node("logic_formalizer",     cancellation_guard(logic_formalizer_node))
    workflow.add_node("logic_direct",         cancellation_guard(logic_direct_node))
    workflow.add_node("logic_solver",         cancellation_guard(logic_solver_node))
    workflow.add_node("logic_explanation",    cancellation_guard(logic_explanation_node))
    workflow.add_node("logic_formalizer_retry", cancellation_guard(_logic_retry_node))

    workflow.add_node("physics_rag",          cancellation_guard(physics_rag_node))
    workflow.add_node("physics_formalizer",   cancellation_guard(physics_formalizer_node))
    workflow.add_node("physics_solver",       cancellation_guard(physics_solver_node))
    workflow.add_node("physics_explanation",  cancellation_guard(physics_explanation_node))
    workflow.add_node("physics_formalizer_retry", cancellation_guard(_physics_retry_node))


    workflow.set_entry_point("classify")


    workflow.add_conditional_edges(
        "classify",
        lambda state: (
            "logic_direct"
            if state.get("task_type") == "logic"
            and should_use_logic_direct(
                state.get("question", ""),
                state.get("options", []),
            )
            else route_after_classify(state)
        ),
        {
            "logic_formalizer": "logic_formalizer",
            "logic_direct":     "logic_direct",
            "physics_rag":      "physics_rag",
        },
    )
    workflow.add_edge("logic_direct", END)

    # Logic branch
    workflow.add_edge("logic_formalizer",  "logic_solver")
    workflow.add_conditional_edges(
        "logic_solver",
        _route_after_logic_solver,
        {
            "logic_formalizer_retry": "logic_formalizer_retry",
            "logic_explanation":      "logic_explanation",
        },
    )
    workflow.add_edge("logic_formalizer_retry", "logic_solver")
    workflow.add_edge("logic_explanation", END)

    # Physics branch
    workflow.add_edge("physics_rag",         "physics_formalizer")
    workflow.add_edge("physics_formalizer",  "physics_solver")
    workflow.add_conditional_edges(
        "physics_solver",
        _route_after_physics_solver,
        {
            "physics_formalizer_retry": "physics_formalizer_retry",
            "physics_explanation":      "physics_explanation",
        },
    )
    workflow.add_edge("physics_formalizer_retry", "physics_solver")
    workflow.add_edge("physics_explanation", END)

    return workflow.compile()



_graph = None


def get_graph():
    global _graph
    if _graph is None:
        logger.info("Compiling LangGraph pipeline...")
        _graph = build_graph()
    return _graph


def run_pipeline(
    question: str,
    premises: list[str] = None,
    collection_name: str = "logic_regulations",
    task_type: Literal["logic", "physics"] | None = None,
    options: list[str] | None = None,
    cancel_event=None,
    deadline: float | None = None,
) -> dict:
    """
    Main entry point for the pipeline.

    Args:
        question: Input question.
        premises: List of premises (for logic tasks).
        collection_name: RAG collection name.
        task_type: Explicit task type from API if provided.

    Returns:
        Dict with answer, explanation, and execution artifacts.
    """
    premises = premises or []
    if premises and task_type in {None, "logic"}:
        retrieved = retrieve_known_logic(question, premises)
        if retrieved is not None:
            logger.info("Answered Type 1 question from disclosed exact-match retrieval.")
            return {"task_type": "logic", **retrieved}

    if not premises and task_type in {None, "physics"}:
        baseline = solve_common_physics(question)
        if baseline is not None:
            logger.info("Solved Type 2 question with deterministic formula baseline.")
            return {"task_type": "physics", **baseline}

    options = options or []
    initial_state: AgentState = {
        "question": question,
        "premises": premises,
        "options": options,
        "task_type": task_type or "logic",
        "requested_task_type": task_type,
        "intermediate_answer": {
            "context_rag": "",
            "context_code": "",
            "generated_code": "",
            "code_output": "",
            "code_error": False,
            "error_message": "",
            "retry_error_feedback": "",
            "reasoning": "",
            "final_output": "",
        },
        "final_answer": {
            "answer": "",
            "explanation": "",
            "fol": "",
            "cot": [],
            "premises": [],
            "premises_used": [],
            "unit": "",
            "confidence": 0.0,
        },
        "error": "",
        "collection_name": collection_name,
        "context": "",
        "retry_count": 0,
    }

    logger.info(f"Processing: {question[:100]}...")
    from src.agent.llm.factory import LLMFactory

    with LLMFactory.pipeline_session(cancel_event), cancellation_context(cancel_event, deadline):
        graph = get_graph()
        result = graph.invoke(initial_state)

    final = result.get("final_answer", {})
    intermediate = result.get("intermediate_answer", {})

    return {
        "task_type":      result.get("task_type"),
        "answer":         final.get("answer"),
        "explanation":    final.get("explanation"),
        "fol":            final.get("fol"),
        "cot":            final.get("cot"),
        "premises":       final.get("premises"),
        "premises_used":  final.get("premises_used", []),
        "unit":           final.get("unit", ""),
        "confidence":     final.get("confidence"),
        "code":           intermediate.get("generated_code"),
        "code_output":    intermediate.get("code_output"),
        "code_error":     intermediate.get("code_error", False),
        "error_message":  intermediate.get("error_message", ""),
        "retry_count":    result.get("retry_count", 0),
    }
