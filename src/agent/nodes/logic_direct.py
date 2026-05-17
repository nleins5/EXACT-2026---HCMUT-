"""
Logic Direct Node — Fallback: LLM giải trực tiếp không qua Z3.
"""
from src.agent.state import AgentState
from src.agent.schema import ExactResponse
from src.utils.logger import logger
from src.agent.prompts.logic import LOGIC_DIRECT_PROMPT


def logic_direct_node(state: AgentState) -> dict:
    """LLM suy luận trực tiếp (dự phòng song song với Z3 pipeline)."""
    try:
        from src.llm.factory import LLMFactory
        llm_client = LLMFactory.create_client(purpose="summary")
        structured_llm = llm_client.get_structured_llm(ExactResponse)

        premises_text = "\n".join(
            [f"- {p}" for p in state.get("premises", [])]
        )
        context_block = (
            f"Premises:\n{premises_text}\n\n" if premises_text else ""
        )

        prompt = LOGIC_DIRECT_PROMPT.format(
            question=f"{context_block}{state['question']}"
        )

        response: ExactResponse = structured_llm.invoke(prompt)
        return {"fallback_answer": response.model_dump()}

    except Exception as e:
        logger.error(f"Lỗi tại logic_direct_node: {e}")
        return {
            "fallback_answer": {
                "answer": "Unknown",
                "explanation": f"Lỗi suy luận trực tiếp: {e}",
                "fol": "",
                "cot": [],
                "premises": [],
                "confidence": 0.0,
            }
        }
