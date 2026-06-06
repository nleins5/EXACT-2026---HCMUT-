"""Physics Explanation Node — tong hop ket qua SymPy thanh ExactResponse (Instruct model).

Tuong tu logic_explanation: 2 nhanh prompt theo `code_error`.
"""
from src.agent.state import AgentState
from src.agent.schema import ExactResponse
from src.agent.nodes.fallbacks import extract_physics_answer, physics_solver_fallback
from src.agent.prompts.physics_explanation import (
    PHYSICS_OUTPUT_PROMPT,
    PHYSICS_OUTPUT_ERROR_PROMPT,
)
from src.agent.runtime import remaining_seconds
from src.core.config import settings
from src.utils.logger import logger


def physics_explanation_node(state: AgentState) -> dict:
    """Sinh ExactResponse JSON tu code_output (success) hoac code (error)."""
    intermediate = state.get("intermediate_answer", {})
    code_error = intermediate.get("code_error", False)

    if remaining_seconds() < settings.api.min_explanation_seconds:
        logger.warning("Skipping physics explanation model because the request budget is low.")
        return physics_solver_fallback(state, "Explanation model skipped due to request budget.")

    try:
        from src.agent.llm.factory import LLMFactory
        llm_client = LLMFactory.activate("instruct")
        structured_llm = llm_client.get_structured_llm(ExactResponse)

        if code_error:
            context = (state.get("context") or "").strip()
            context_block = (
                f"Relevant Formulas/Context:\n{context}\n" if context else ""
            )
            prompt = PHYSICS_OUTPUT_ERROR_PROMPT.format(
                question=state["question"],
                context_block=context_block,
                generated_code=intermediate.get("generated_code", ""),
                error_message=intermediate.get("error_message", ""),
            )
            logger.info("Physics explanation: dung ERROR prompt (code loi).")
        else:
            prompt = PHYSICS_OUTPUT_PROMPT.format(
                question=state["question"],
                generated_code=intermediate.get("generated_code", ""),
                code_output=intermediate.get("code_output", ""),
            )
            logger.info("Physics explanation: dung SUCCESS prompt.")

        response: ExactResponse = structured_llm.invoke(prompt)
        final_answer = response.model_dump()
        if not code_error:
            verified_answer = extract_physics_answer(intermediate.get("code_output", ""))
            if verified_answer:
                final_answer["answer"] = verified_answer
                final_answer["confidence"] = max(
                    float(final_answer.get("confidence") or 0.0),
                    0.8,
                )
        return {"final_answer": final_answer}

    except Exception as e:
        logger.error(f"physics_explanation_node loi: {e}")
        return physics_solver_fallback(state, str(e))
