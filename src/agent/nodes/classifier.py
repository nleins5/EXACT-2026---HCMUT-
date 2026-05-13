"""
Classifier Node
Phân loại câu hỏi thành: logic (quy chế) hoặc physics (vật lý)
"""
from src.agent.state import AgentState
from src.utils.logger import logger
from src.prompt.templete import CLASSIFY_PROMPT


def classify_node(state: AgentState) -> dict:
    """
    Phân loại câu hỏi dựa trên cấu trúc đầu vào của EXACT 2026:
    - Type 1 (Logic): Có đi kèm danh sách giả thiết (premises).
    - Type 2: Không có premises (Có thể là Logic hoặc Physics).
    """
    premises = state.get("premises", [])
    
    if premises:
        task_type = "logic"
        logger.info(f"Phát hiện {len(premises)} giả thiết, phân loại là: logic (Type 1)")
    else:
        # Nếu không có premises, dùng LLM để phân biệt Logic vs Physics
        task_type = _llm_classify(state["question"])
        logger.info(f"Phân loại bởi LLM: {task_type} (Type 2)")
    
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
        logger.warning(f"Bộ phân loại LLM thất bại, mặc định chọn 'logic': {e}")
        return "logic"


def route_after_classify(state: AgentState) -> list[str]:
    """
    Hàm Router — quyết định nhánh nào của graph sẽ được thực thi 
    dựa trên kết quả phân loại (task_type).
    Trả về danh sách các node để hỗ trợ chạy song song (fan-out).
    """
    t_type = state.get("task_type", "logic")
    if t_type == "physics":
        return ["physics_rag"]
    
    # Mặc định là logic, chạy song song 2 nhánh
    return ["logic_formalizer", "logic_direct"]
