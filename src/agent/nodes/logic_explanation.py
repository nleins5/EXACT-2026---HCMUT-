"""Logic Explanation Node — tong hop ket qua Z3 thanh ExactResponse (Instruct model).

Hai nhanh prompt mirror dataset instruct.jsonl:
- code_error == False -> SUCCESS prompt (tin code_output).
- code_error == True  -> ERROR prompt (doc code lam hint).

Bat ke nhanh nao deu goi LLM dung 1 lan -> dam bao timing 60s.
"""
from src.agent.state import AgentState
from src.agent.schema import ExactResponse
from src.agent.nodes.fallbacks import logic_solver_fallback, normalize_logic_answer
from src.agent.prompts.logic_explanation import (
    LOGIC_OUTPUT_PROMPT,
    LOGIC_OUTPUT_ERROR_PROMPT,
)
from src.agent.runtime import remaining_seconds
from src.core.config import settings
from src.utils.logger import logger
from src.utils.z3_output_parser import parse_z3_output


def logic_explanation_node(state: AgentState) -> dict:
    """Sinh ExactResponse JSON tu code_output (success) hoac code (error)."""
    intermediate = state.get("intermediate_answer", {})
    code_error = intermediate.get("code_error", False)

    if not code_error:
        return logic_solver_fallback(state, "")

    if remaining_seconds() < settings.api.min_explanation_seconds:
        logger.warning("Skipping logic explanation model because the request budget is low.")
        return logic_solver_fallback(state, "Explanation model skipped due to request budget.")

    try:
        from src.agent.llm.factory import LLMFactory
        llm_client = LLMFactory.activate("instruct")
        structured_llm = llm_client.get_structured_llm(ExactResponse)

        if code_error:
            premises = state.get("premises", []) or []
            premises_block = (
                "\n".join([f"- {p}" for p in premises]) or "(none)"
            )
            prompt = LOGIC_OUTPUT_ERROR_PROMPT.format(
                question=state["question"],
                premises_block=premises_block,
                generated_code=intermediate.get("generated_code", ""),
                error_message=intermediate.get("error_message", ""),
            )
            logger.info("Logic explanation: dung ERROR prompt (code loi).")
        else:
            prompt = LOGIC_OUTPUT_PROMPT.format(
                question=state["question"],
                generated_code=intermediate.get("generated_code", ""),
                code_output=intermediate.get("code_output", ""),
            )
            logger.info("Logic explanation: dung SUCCESS prompt.")

        response: ExactResponse = structured_llm.invoke(prompt)
        final_answer = response.model_dump()
        if not code_error:
            verified_answer = normalize_logic_answer(parse_z3_output(intermediate.get("code_output", "")))
            if verified_answer != "Unknown":
                final_answer["answer"] = verified_answer
                final_answer["confidence"] = max(
                    float(final_answer.get("confidence") or 0.0),
                    0.85,
                )
        return {"final_answer": final_answer}

    except Exception as e:
        logger.error(f"logic_explanation_node loi: {e}")
        return logic_solver_fallback(state, str(e))
