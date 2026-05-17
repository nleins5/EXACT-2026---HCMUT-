"""
Physics Formalizer Node — Dịch bài toán vật lý sang mã SymPy.
"""
import re
from src.agent.state import AgentState
from src.utils.logger import logger
from src.agent.prompts.physics import PHYSICS_SYSTEM_PROMPT


def physics_formalizer_node(state: AgentState) -> dict:
    """Dịch câu hỏi vật lý + ngữ cảnh RAG thành mã SymPy."""
    try:
        from src.llm.factory import LLMFactory
        llm = LLMFactory.create_client(purpose="reasoning").get_llm()

        context = state.get("context", "")
        context_block = (
            f"\n\nRelevant Formulas/Context:\n{context}\n"
            if context else ""
        )

        user_prompt = f"""{context_block}
Problem:
{state['question']}

Generate Python code using SymPy to solve this.
Requirements:
1. Define symbols for all physical quantities.
2. Use SI units.
3. Print steps and the final result as: print(f"FINAL_ANSWER: {{value}} {{unit}}")
"""
        from langchain_core.messages import SystemMessage, HumanMessage
        response = llm.invoke([
            SystemMessage(content=PHYSICS_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])

        code = _extract_code(response.content)
        logger.info("Đã dịch bài toán vật lý sang mã SymPy.")

        intermediate = state.get("intermediate_answer", {})
        intermediate["generated_code"] = code
        return {"intermediate_answer": intermediate}

    except Exception as e:
        logger.error(f"Lập trình hóa vật lý thất bại: {e}")
        intermediate = state.get("intermediate_answer", {})
        intermediate["generated_code"] = ""
        return {"intermediate_answer": intermediate, "error": str(e)}


def _extract_code(text: str) -> str:
    """Trích xuất mã Python từ phản hồi của LLM."""
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()
