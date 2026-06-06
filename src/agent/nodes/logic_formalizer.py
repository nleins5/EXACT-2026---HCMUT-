"""Logic Formalizer Node — sinh code Z3 cho bai toan logic (Coder model)."""
from src.agent.state import AgentState
from src.agent.prompts.logic_formalizer import Z3_SYSTEM_PROMPT, Z3_USER_TEMPLATE
from src.utils.code_extract import extract_python_code
from src.utils.logger import logger


def logic_formalizer_node(state: AgentState) -> dict:
    """Dich cau hoi logic + premises sang code Z3.

    Su dung Coder model (qwen2.5-coder-7b-instruct). Goi
    `LLMFactory.activate("coder")` -> swap process neu can (BTC Q3).
    """
    try:
        from src.agent.llm.factory import LLMFactory
        llm = LLMFactory.activate("coder").get_llm()

        premises = state.get("premises", []) or []
        if premises:
            premises_text = "\n".join([f"- {p}" for p in premises])
            premises_block = f"Premises:\n{premises_text}\n\n"
        else:
            premises_block = ""

        intermediate = state.get("intermediate_answer", {})
        user_prompt = Z3_USER_TEMPLATE.format(
            premises_block=premises_block,
            question=state["question"],
        )
        retry_feedback = (intermediate.get("retry_error_feedback") or "").strip()
        if retry_feedback:
            user_prompt += (
                "\n\nPrevious Z3 code failed with this runtime/syntax error:\n"
                f"{retry_feedback}\n"
                "Regenerate corrected code. Keep the same output format."
            )

        from langchain_core.messages import SystemMessage, HumanMessage
        response = llm.invoke([
            SystemMessage(content=Z3_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])

        code = extract_python_code(response.content)
        if code:
            logger.info(f"Logic formalizer: sinh duoc code Z3 ({len(code)} chars).")
        else:
            logger.warning("Logic formalizer: khong trich xuat duoc code Python.")

        intermediate["generated_code"] = code
        return {"intermediate_answer": intermediate}

    except Exception as e:
        logger.error(f"logic_formalizer_node loi: {e}")
        intermediate = state.get("intermediate_answer", {})
        intermediate["generated_code"] = ""
        return {"intermediate_answer": intermediate, "error": str(e)}
