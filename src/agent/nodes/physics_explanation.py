"""
Physics Explanation Node — Tổng hợp kết quả SymPy thành ExactResponse.

Có 2 nhánh prompt dựa trên `intermediate.code_error`:
- code_error == False  → dùng PHYSICS_OUTPUT_PROMPT (tin tưởng output SymPy).
- code_error == True   → dùng PHYSICS_OUTPUT_ERROR_PROMPT (đọc code lỗi + RAG
                          context như gợi ý, để LLM tự suy luận ra đáp án).
"""
from src.agent.state import AgentState
from src.agent.schema import ExactResponse
from src.utils.logger import logger
from src.agent.prompts.physics import (
    PHYSICS_OUTPUT_PROMPT,
    PHYSICS_OUTPUT_ERROR_PROMPT,
)


def physics_explanation_node(state: AgentState) -> dict:
    """Tổng hợp kết quả SymPy (hoặc fallback từ code lỗi) thành ExactResponse."""
    intermediate = state.get("intermediate_answer", {})
    code_error = intermediate.get("code_error", False)

    try:
        from src.llm.factory import LLMFactory
        llm_client = LLMFactory.create_client(purpose="summary")
        structured_llm = llm_client.get_structured_llm(ExactResponse)

        if code_error:
            context = state.get("context", "")
            context_block = (
                f"Relevant Formulas/Context:\n{context}\n" if context else ""
            )
            prompt = PHYSICS_OUTPUT_ERROR_PROMPT.format(
                question=state["question"],
                context_block=context_block,
                generated_code=intermediate.get("generated_code", ""),
                error_message=intermediate.get("error_message", ""),
            )
            logger.info("Physics explanation: dùng ERROR prompt (code lỗi).")
        else:
            prompt = PHYSICS_OUTPUT_PROMPT.format(
                question=state["question"],
                code_output=intermediate.get("code_output", ""),
            )
            logger.info("Physics explanation: dùng SUCCESS prompt.")

        response: ExactResponse = structured_llm.invoke(prompt)
        return {"final_answer": response.model_dump()}

    except Exception as e:
        logger.error(f"Lỗi tại physics_explanation_node: {e}")
        return {
            "final_answer": {
                "answer": "Error",
                "explanation": f"Lỗi hệ thống: {e}",
                "fol": "",
                "cot": [],
                "premises": [],
                "confidence": 0.0,
            },
            "error": str(e),
        }
