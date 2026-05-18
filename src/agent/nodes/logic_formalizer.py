"""
Logic Formalizer Node — Dịch bài toán logic sang mã Z3-Python.
"""
from src.agent.state import AgentState
from src.utils.logger import logger
from src.utils.code_extract import extract_python_code
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
Define variables for each entity and add constraints for each premise.
Output ONLY one ```python ... ``` fenced block. No prose, no <think>.
"""

        from langchain_core.messages import SystemMessage, HumanMessage
        response = llm.invoke([
            SystemMessage(content=Z3_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])

        code = extract_python_code(response.content)
        if code:
            logger.info(f"Đã dịch bài toán sang mã Z3 ({len(code)} chars).")
        else:
            logger.warning("Logic formalizer: không trích xuất được code Python từ LLM output.")

        intermediate = state.get("intermediate_answer", {})
        intermediate["generated_code"] = code
        return {"intermediate_answer": intermediate}

    except Exception as e:
        logger.error(f"Dịch logic thất bại: {e}")
        intermediate = state.get("intermediate_answer", {})
        intermediate["generated_code"] = ""
        return {"intermediate_answer": intermediate, "error": str(e)}
