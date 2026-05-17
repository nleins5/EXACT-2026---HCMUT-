"""
Physics Direct Node — Fallback: LLM giải vật lý trực tiếp không qua SymPy.
"""
from src.agent.state import AgentState
from src.agent.schema import ExactResponse
from src.utils.logger import logger
from src.agent.prompts.physics import PHYSICS_DIRECT_PROMPT


def physics_direct_node(state: AgentState) -> dict:
    """LLM suy luận trực tiếp (dự phòng song song với SymPy pipeline)."""
    try:
        from src.llm.factory import LLMFactory
        llm_client = LLMFactory.create_client(purpose="summary")
        structured_llm = llm_client.get_structured_llm(ExactResponse)

        context = state.get("context", "")
        context_block = (
            f"\n\nKnowledge Context:\n{context}\n" if context else ""
        )

        prompt = PHYSICS_DIRECT_PROMPT.format(
            question=state["question"],
            context_block=context_block,
        )

        response: ExactResponse = structured_llm.invoke(prompt)
        return {"fallback_answer": response.model_dump()}

    except Exception as e:
        logger.error(f"Lỗi tại physics_direct_node: {e}")
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
