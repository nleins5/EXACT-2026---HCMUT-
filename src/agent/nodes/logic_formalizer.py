"""
Logic Formalizer Node — Dịch bài toán logic sang mã Z3-Python.
"""
import re
from src.agent.state import AgentState
from src.utils.logger import logger
from src.agent.prompts.logic import Z3_SYSTEM_PROMPT


def logic_formalizer_node(state: AgentState) -> dict:
    """Dịch câu hỏi ngôn ngữ tự nhiên sang mã Python Z3."""
    try:
        from src.llm.factory import LLMFactory
        llm = LLMFactory.create_client(purpose="reasoning").get_llm()

        premises_text = "\n".join(
            [f"- {p}" for p in state.get("premises", [])]
        )
        context_block = (
            f"Premises:\n{premises_text}\n\n" if premises_text else ""
        )

        user_prompt = f"""{context_block}Logic Problem:
{state['question']}

Translate the logic problem above into Python Z3 code.
Ensure you define variables for each entity and add constraints for each premise.
"""

        from langchain_core.messages import SystemMessage, HumanMessage
        response = llm.invoke([
            SystemMessage(content=Z3_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])

        code = _extract_code(response.content)
        logger.info("Đã dịch bài toán sang mã Z3.")

        intermediate = state.get("intermediate_answer", {})
        intermediate["generated_code"] = code
        return {"intermediate_answer": intermediate}

    except Exception as e:
        logger.error(f"Dịch logic thất bại: {e}")
        intermediate = state.get("intermediate_answer", {})
        intermediate["generated_code"] = ""
        return {"intermediate_answer": intermediate, "error": str(e)}


def _extract_code(text: str) -> str:
    """Trích xuất mã Python từ phản hồi của LLM."""
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()
