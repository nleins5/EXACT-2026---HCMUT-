"""
Physics Node — Formalization, Solver, Explanation
1. RAG Node: Tìm kiếm công thức liên quan đến "điện trở", "tụ điện", "định luật Ohm"... dựa trên câu hỏi.
2. Formalizer Node: LLM nhận câu hỏi + công thức từ RAG để viết code tính toán (SymPy).
3. Solver Node: Chạy code để lấy con số cuối cùng (Numerical Solver).
4. Explanation Node: Đóng gói thành định dạng Chain-of-Thought (CoT).
5. Direct Node (Parallel Fallback): Giải trực tiếp bằng LLM như một phương án dự phòng.
"""
import re
import subprocess
import sys
from src.agent.state import AgentState
from src.agent.schema import ExactResponse
from src.utils.logger import logger
from src.prompt.templete import PHYSICS_SYSTEM_PROMPT, PHYSICS_OUTPUT_PROMPT, PHYSICS_DIRECT_PROMPT


def physics_rag_node(state: AgentState) -> AgentState:
    """Node 1: Physics Context (RAG - Truy xuất kiến thức).
    
    Tìm kiếm các công thức vật lý, hằng số và ngữ cảnh liên quan từ cơ sở tri thức (Vector DB).
    Mục tiêu là cung cấp đủ "nguyên liệu" cho LLM để giải bài toán chính xác.

    Args:
        state: Trạng thái hiện tại chứa câu hỏi.

    Returns:
        Trạng thái đã cập nhật trường `context` chứa thông tin truy xuất được.
    """
    try:
        from src.retrieval.engine import Retriever
        retriever = Retriever()
        docs = retriever.retrieval(
            query=state["question"],
            collection_name="physics_knowledge",
            mode="hybrid",
        )
        if docs:
            context = "\n\n".join([d.node.get_content() for d in docs])
            logger.info(f"Physics RAG: Đã tìm thấy {len(docs)} tài liệu liên quan.")
        else:
            context = "Không tìm thấy công thức cụ thể trong cơ sở tri thức."
            logger.info("Physics RAG: Không tìm thấy tài liệu nào.")
    except Exception as e:
        logger.warning(f"Physics RAG thất bại: {e}")
        context = ""

    return {"context": context}


def physics_formalizer_node(state: AgentState) -> AgentState:
    """Node 2: Physics Formalizer (Lập trình hóa bài toán).
    
    Dịch câu hỏi vật lý và ngữ cảnh từ RAG thành mã Python sử dụng thư viện SymPy.
    SymPy giúp giải các phương trình đại số và đơn vị một cách chính xác.

    Args:
        state: Trạng thái chứa câu hỏi và ngữ cảnh RAG.

    Returns:
        Trạng thái đã cập nhật mã code SymPy trong `intermediate_answer`.
    """
    try:
        from src.llm.factory import LLMFactory
        llm = LLMFactory.create_client(purpose="reasoning").get_llm()

        context = state.get("context", "")
        context_block = f"\n\nRelevant Formulas/Context:\n{context}\n" if context else ""

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


def physics_solver_node(state: AgentState) -> AgentState:
    """Node 3: Numerical Solver (Bộ giải số học).
    
    Thực thi mã SymPy đã sinh ra để tính toán kết quả số học cuối cùng.

    Args:
        state: Trạng thái chứa mã code cần giải.

    Returns:
        Trạng thái cập nhật kết quả in ra của mã code (stdout).
    """
    intermediate = state.get("intermediate_answer", {})
    code = intermediate.get("generated_code", "")
    if not code:
        intermediate["code_output"] = "LỖI: Không có mã code để thực thi"
        return {"intermediate_answer": intermediate}

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=30
        )
        output = result.stdout.strip() if result.returncode == 0 else f"LỖI THỰC THI (RUNTIME ERROR):\n{result.stderr}"
        logger.info(f"Kết quả Solver vật lý: {output[:200]}")
        intermediate["code_output"] = output
        return {"intermediate_answer": intermediate}
    except Exception as e:
        logger.error(f"Thực thi Solver vật lý thất bại: {e}")
        intermediate["code_output"] = f"LỖI: {e}"
        return {"intermediate_answer": intermediate}


def physics_explanation_node(state: AgentState) -> dict:
    """Node 4: Explanation & Formatting (Giải thích & Trình bày).
    
    Sử dụng LLM với Structured Output để tổng hợp kết quả tính toán 
    thành định dạng chuẩn của cuộc thi EXACT 2026.
    """
    intermediate = state.get("intermediate_answer", {})
    code_output = intermediate.get("code_output", "")
    
    try:
        from src.llm.factory import LLMFactory
        llm_client = LLMFactory.create_client(purpose="summary")
        structured_llm = llm_client.get_structured_llm(ExactResponse)
        
        prompt = PHYSICS_OUTPUT_PROMPT.format(
            question=state["question"],
            code_output=code_output
        )
        
        # Gọi LLM và nhận trực tiếp object ExactResponse
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
                "confidence": 0.0
            }
        }


def physics_direct_node(state: AgentState) -> dict:
    """Nhánh song song: LLM suy luận trực tiếp (Dự phòng).
    
    Sử dụng LLM với Structured Output để giải bài toán trực tiếp 
    không qua bộ giải SymPy.
    """
    try:
        from src.llm.factory import LLMFactory
        llm_client = LLMFactory.create_client(purpose="summary")
        structured_llm = llm_client.get_structured_llm(ExactResponse)
        
        context = state.get("context", "")
        context_block = f"\n\nKnowledge Context:\n{context}\n" if context else ""

        prompt = PHYSICS_DIRECT_PROMPT.format(
            question=state["question"],
            context_block=context_block
        )
        
        # Gọi LLM và nhận object ExactResponse
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
                "confidence": 0.0
            }
        }


def _extract_code(text: str) -> str:
    """Trích xuất mã Python từ phản hồi của LLM."""
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()
