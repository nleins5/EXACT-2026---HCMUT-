"""
Logic Explanation Node — Tổng hợp kết quả thành ExactResponse.

Có 2 nhánh prompt dựa trên `intermediate.code_error`:
- code_error == False  → dùng LOGIC_OUTPUT_PROMPT (tin tưởng output Z3).
- code_error == True   → dùng LOGIC_OUTPUT_ERROR_PROMPT (đọc code lỗi như gợi ý
                          rồi để LLM tự suy luận ra đáp án).

Cả 2 nhánh đều gọi LLM đúng 1 lần để tránh tràn RAM và đảm bảo timing < 60s.
"""
from src.agent.state import AgentState
from src.agent.schema import ExactResponse
from src.utils.logger import logger
from src.agent.prompts.logic import (
    LOGIC_OUTPUT_PROMPT,
    LOGIC_OUTPUT_ERROR_PROMPT,
)


def logic_explanation_node(state: AgentState) -> dict:
    """Tổng hợp kết quả Z3 (hoặc fallback từ code lỗi) thành ExactResponse."""
    intermediate = state.get("intermediate_answer", {})
    code_error = intermediate.get("code_error", False)

    try:
        from src.llm.factory import LLMFactory
        llm_client = LLMFactory.create_client(purpose="summary")
        structured_llm = llm_client.get_structured_llm(ExactResponse)

        if code_error:
            premises_block = "\n".join(
                [f"- {p}" for p in state.get("premises", [])]
            ) or "(none)"

            prompt = LOGIC_OUTPUT_ERROR_PROMPT.format(
                question=state["question"],
                premises_block=premises_block,
                generated_code=intermediate.get("generated_code", ""),
                error_message=intermediate.get("error_message", ""),
            )
            logger.info("Logic explanation: dùng ERROR prompt (code lỗi).")
        else:
            prompt = LOGIC_OUTPUT_PROMPT.format(
                question=state["question"],
                code_output=intermediate.get("code_output", ""),
            )
            logger.info("Logic explanation: dùng SUCCESS prompt.")

        response: ExactResponse = structured_llm.invoke(prompt)
        return {"final_answer": response.model_dump()}

    except Exception as e:
        logger.error(f"Lỗi tại logic_explanation_node: {e}")
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
