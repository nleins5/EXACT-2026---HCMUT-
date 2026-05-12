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


def physics_explanation_node(state: AgentState) -> AgentState:
    """Node 4: Explanation & Formatting (Giải thích & Trình bày).
    
    Chuyển đổi kết quả tính toán khô khan thành lời giải Chain-of-Thought (CoT) mượt mà.
    Nếu bộ giải SymPy thất bại, node này sẽ lấy kết quả từ nhánh suy luận trực tiếp (fallback).

    Args:
        state: Trạng thái chứa kết quả tính toán và dự phòng.

    Returns:
        Trạng thái cập nhật đáp án cuối cùng (final_answer).
    """
    try:
        from src.llm.factory import LLMFactory
        llm = LLMFactory.create_client(purpose="reasoning").get_llm()

        intermediate = state.get("intermediate_answer", {})
        code_output = intermediate.get("code_output", "")
        fallback = state.get("fallback_answer", {})

        # Kiểm tra tính thành công của code output
        success = "FINAL_ANSWER:" in code_output
        
        if success:
            prompt = PHYSICS_OUTPUT_PROMPT.format(
                question=state['question'],
                code_output=code_output
            )
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content=prompt)])
            answer, reasoning = _parse_output(response.content, code_output)
            logger.info("Đã tạo lời giải vật lý dựa trên kết quả Solver.")
        else:
            # Nhánh dự phòng khi Solver thất bại
            answer = fallback.get("answer", "Không xác định")
            reasoning = fallback.get("reasoning", "Bộ giải toán ký hiệu thất bại. Đang sử dụng suy luận trực tiếp từ LLM.")
            logger.warning("Solver vật lý thất bại, sử dụng đáp án dự phòng từ LLM.")

        final = {
            "answer": answer,
            "reasoning": reasoning,
            "final_output": f"Lập luận:\n{reasoning}\n\nĐáp án cuối cùng:\n{answer}"
        }
        return {"final_answer": final}

    except Exception as e:
        logger.error(f"Tạo giải thích vật lý thất bại: {e}")
        return state


def physics_direct_node(state: AgentState) -> AgentState:
    """Parallel Node: Direct LLM Reasoning (Suy luận trực tiếp - Nhánh song song).
    
    Chạy song song với nhánh Formalizer/Solver để cung cấp một đáp án cơ sở.
    Đóng vai trò là phương án dự phòng an toàn (Safety Net).

    Args:
        state: Trạng thái chứa câu hỏi và ngữ cảnh RAG.

    Returns:
        Trạng thái cập nhật kết quả dự phòng vào `fallback_answer`.
    """
    try:
        from src.llm.factory import LLMFactory
        llm = LLMFactory.create_client(purpose="reasoning").get_llm()

        context = state.get("context", "")
        context_block = f"\n\nKnowledge Context:\n{context}\n" if context else ""

        prompt = PHYSICS_DIRECT_PROMPT.format(
            question=state["question"],
            context_block=context_block
        )
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=prompt)])
        
        answer, reasoning = _parse_direct_output(response.content)
        logger.info("Đã tạo đáp án vật lý suy luận trực tiếp (nhánh song song).")
        
        fallback = {
            "answer": answer,
            "reasoning": reasoning,
            "final_output": f"Lập luận:\n{reasoning}\n\nĐáp án cuối cùng:\n{answer}"
        }
        return {"fallback_answer": fallback}
    except Exception as e:
        logger.error(f"Nhánh suy luận vật lý trực tiếp thất bại: {e}")
        return state


def _extract_code(text: str) -> str:
    """Extracts Python code block from LLM text."""
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _extract_final_answer(output: str) -> str:
    """Extracts numeric answer from stdout."""
    match = re.search(r"FINAL_ANSWER:\s*(.+)", output)
    return match.group(1).strip() if match else "Unknown"


def _parse_output(content: str, fallback_out: str) -> tuple[str, str]:
    """Parses Answer and Reasoning from structured text."""
    reasoning = ""
    answer = ""

    reasoning_match = re.search(r"Reasoning:\s*\n?(.*?)(?:\n\nFinal Answer:|\Z)", content, re.DOTALL)
    answer_match = re.search(r"Final Answer:\s*\n?(.*)", content, re.DOTALL)

    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    else:
        reasoning = content

    if answer_match:
        answer = answer_match.group(1).strip()
    elif fallback_out:
        answer = _extract_final_answer(fallback_out)

    return answer, reasoning


def _parse_direct_output(content: str) -> tuple[str, str]:
    """Specialized parser for direct LLM reasoning output."""
    reasoning = ""
    answer = ""
    
    # Try to find Reasoning block
    reasoning_match = re.search(r"Reasoning:\s*\n?(.*?)(?:\n\nFinal Answer:|\Z)", content, re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    else:
        reasoning = content

    # Try to find Final Answer block
    answer_match = re.search(r"Final Answer:\s*\n?(.*)", content, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
    
    return answer, reasoning
