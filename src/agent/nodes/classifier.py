"""
Classifier Node — Phân loại câu hỏi thành logic hoặc physics.
"""
from src.agent.state import AgentState
from src.utils.logger import logger
from src.agent.prompts.classify import CLASSIFY_PROMPT


def classify_node(state: AgentState) -> dict:
    """
    Phân loại câu hỏi:
    - Type 1 (Logic): Có premises đi kèm.
    - Type 2: Không có premises → dùng LLM phân loại.
    """
    premises = state.get("premises", [])

    if premises:
        task_type = "logic"
        logger.info(f"Phát hiện {len(premises)} giả thiết → logic (Type 1)")
    else:
        task_type = _llm_classify(state["question"])
        logger.info(f"LLM phân loại: {task_type} (Type 2)")

    return {"task_type": task_type}


def _llm_classify(question: str) -> str:
    """Sử dụng LLM để phân loại trong các trường hợp mơ hồ."""
    try:
        from src.llm.factory import LLMFactory
        llm = LLMFactory.create_client(purpose="reasoning")

        prompt = CLASSIFY_PROMPT.format(question=question)
        response = llm.get_llm().invoke(prompt)
        result = response.content.strip().lower()

        if "physics" in result:
            return "physics"
        return "logic"
    except Exception as e:
        logger.warning(f"LLM classifier thất bại, mặc định 'logic': {e}")
        return "logic"


def route_after_classify(state: AgentState) -> list[str]:
    """
    Router — quyết định nhánh xử lý dựa trên task_type.
    Trả về danh sách node để hỗ trợ fan-out (chạy song song).
    """
    t_type = state.get("task_type", "logic")
    if t_type == "physics":
        return ["physics_rag"]

    return ["logic_formalizer", "logic_direct"]
