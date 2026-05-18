"""Logic Explanation Node — tong hop ket qua Z3 thanh ExactResponse (Instruct model).

Hai nhanh prompt mirror dataset instruct.jsonl:
- code_error == False -> SUCCESS prompt (tin code_output).
- code_error == True  -> ERROR prompt (doc code lam hint).

Bat ke nhanh nao deu goi LLM dung 1 lan -> dam bao timing 60s.
"""
from src.agent.state import AgentState
from src.agent.schema import ExactResponse
from src.agent.prompts.logic_explanation import (
    LOGIC_OUTPUT_PROMPT,
    LOGIC_OUTPUT_ERROR_PROMPT,
)
from src.utils.logger import logger


def logic_explanation_node(state: AgentState) -> dict:
    """Sinh ExactResponse JSON tu code_output (success) hoac code (error)."""
    intermediate = state.get("intermediate_answer", {})
    code_error = intermediate.get("code_error", False)

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
                code_output=intermediate.get("code_output", ""),
            )
            logger.info("Logic explanation: dung SUCCESS prompt.")

        response: ExactResponse = structured_llm.invoke(prompt)
        return {"final_answer": response.model_dump()}

    except Exception as e:
        logger.error(f"logic_explanation_node loi: {e}")
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
