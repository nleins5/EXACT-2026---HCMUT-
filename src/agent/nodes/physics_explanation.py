"""Physics Explanation Node — tong hop ket qua SymPy thanh ExactResponse (Instruct model).

Tuong tu logic_explanation: 2 nhanh prompt theo `code_error`.
"""
from src.agent.state import AgentState
from src.agent.schema import ExactResponse
from src.agent.prompts.physics_explanation import (
    PHYSICS_OUTPUT_PROMPT,
    PHYSICS_OUTPUT_ERROR_PROMPT,
)
from src.utils.logger import logger


def physics_explanation_node(state: AgentState) -> dict:
    """Sinh ExactResponse JSON tu code_output (success) hoac code (error)."""
    intermediate = state.get("intermediate_answer", {})
    code_error = intermediate.get("code_error", False)

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
        return {"final_answer": response.model_dump()}

    except Exception as e:
        logger.error(f"physics_explanation_node loi: {e}")
        return {
            "final_answer": {
                "answer": "Error",
                "explanation": f"Loi he thong: {e}",
                "fol": "",
                "cot": [],
                "premises": [],
                "confidence": 0.0,
            },
            "error": str(e),
        }
