"""Physics Formalizer Node — sinh code SymPy cho bai toan vat ly (Coder model).

Khac voi logic: nhan them `state["context"]` tu RAG node (neu co), chen vao
prompt nhu `context_block` chua cong thuc/vi du SymPy lien quan.
"""
from src.agent.state import AgentState
from src.agent.prompts.physics_formalizer import (
    PHYSICS_SYSTEM_PROMPT,
    PHYSICS_USER_TEMPLATE,
)
from src.utils.code_extract import extract_python_code
from src.utils.logger import logger


def physics_formalizer_node(state: AgentState) -> dict:
    """Dich cau hoi vat ly + ngu canh RAG sang code SymPy.

    Su dung Coder model (qwen2.5-coder-7b-instruct). Goi
    `LLMFactory.activate("coder")` -> swap process neu can (BTC Q3).
    """
    try:
        from src.agent.llm.factory import LLMFactory
        llm = LLMFactory.activate("coder").get_llm()

        context = (state.get("context") or "").strip()
        if context:
            context_block = (
                f"Relevant Formulas/Examples:\n{context}\n\n"
            )
        else:
            context_block = ""

        user_prompt = PHYSICS_USER_TEMPLATE.format(
            context_block=context_block,
            question=state["question"],
        )

        from langchain_core.messages import SystemMessage, HumanMessage
        response = llm.invoke([
            SystemMessage(content=PHYSICS_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])

        code = extract_python_code(response.content)
        if code:
            logger.info(f"Physics formalizer: sinh duoc code SymPy ({len(code)} chars).")
        else:
            logger.warning("Physics formalizer: khong trich xuat duoc code Python.")

        intermediate = state.get("intermediate_answer", {})
        intermediate["generated_code"] = code
        return {"intermediate_answer": intermediate}

    except Exception as e:
        logger.error(f"physics_formalizer_node loi: {e}")
        intermediate = state.get("intermediate_answer", {})
        intermediate["generated_code"] = ""
        return {"intermediate_answer": intermediate, "error": str(e)}
