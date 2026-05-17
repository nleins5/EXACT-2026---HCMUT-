"""
Physics Explanation Node — Tổng hợp kết quả SymPy thành ExactResponse.
"""
from src.agent.state import AgentState
from src.agent.schema import ExactResponse
from src.utils.logger import logger
from src.agent.prompts.physics import PHYSICS_OUTPUT_PROMPT


def physics_explanation_node(state: AgentState) -> dict:
    """Tổng hợp kết quả tính toán thành định dạng EXACT 2026."""
    intermediate = state.get("intermediate_answer", {})
    code_output = intermediate.get("code_output", "")

    try:
        from src.llm.factory import LLMFactory
        llm_client = LLMFactory.create_client(purpose="summary")
        structured_llm = llm_client.get_structured_llm(ExactResponse)

        prompt = PHYSICS_OUTPUT_PROMPT.format(
            question=state["question"],
            code_output=code_output,
        )

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
            }
        }
