"""
Logic Node — Formalization, Solver, Explanation
1. Formalization: LLM đóng vai trò Logic Translator dịch bài toán sang Z3-Python.
2. Solver: Thực thi code Z3 (Symbolic Solver).
3. Explanation: Tổng hợp kết quả và giải thích (Human-readable explanation).
"""
import re
import subprocess
import sys
from src.agent.state import AgentState
from src.utils.logger import logger
from src.prompt.templete import Z3_SYSTEM_PROMPT, LOGIC_OUTPUT_PROMPT, LOGIC_DIRECT_PROMPT


def logic_formalizer_node(state: AgentState) -> AgentState:
    """Node 1: Logic Formalizer (Dịch sang Logic hình thức).
    
    Sử dụng LLM để dịch câu hỏi ngôn ngữ tự nhiên sang mã Python Z3.
    Mục tiêu là mô hình hóa các thực thể và ràng buộc logic của bài toán.

    Args:
        state: Trạng thái hiện tại chứa câu hỏi.

    Returns:
        Trạng thái cập nhật với mã Z3 trong `intermediate_answer`.
    """
    try:
        from src.llm.factory import LLMFactory
        llm = LLMFactory.create_client(purpose="reasoning").get_llm()

        # Nhánh logic chỉ tập trung vào việc dịch mã, không dùng RAG theo yêu cầu
        premises_text = "\n".join([f"- {p}" for p in state.get("premises", [])])
        context_block = f"Premises:\n{premises_text}\n\n" if premises_text else ""
        
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

        # Trích xuất khối mã code từ phản hồi của LLM
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


def logic_solver_node(state: AgentState) -> AgentState:
    """Node 2: Symbolic Solver (Bộ giải ký hiệu).
    
    Thực thi mã Z3 đã sinh ra trong một tiến trình con (subprocess) an toàn.
    Kết quả trả về thường là 'sat' kèm theo model hoặc kết quả in ra ANSWER: <đáp án>.

    Args:
        state: Trạng thái chứa mã Z3 cần giải.

    Returns:
        Trạng thái cập nhật kết quả thực thi mã.
    """
    intermediate = state.get("intermediate_answer", {})
    code = intermediate.get("generated_code", "")
    if not code:
        intermediate["code_output"] = "LỖI: Không có mã code để thực thi"
        return {**state, "intermediate_answer": intermediate}

    try:
        # Chạy code Python trong subprocess với timeout 30s
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=30
        )
        output = result.stdout.strip() if result.returncode == 0 else f"LỖI THỰC THI (RUNTIME ERROR):\n{result.stderr}"
        logger.info(f"Kết quả Solver: {output[:200]}")
        intermediate["code_output"] = output
        return {"intermediate_answer": intermediate}
    except subprocess.TimeoutExpired:
        intermediate["code_output"] = "LỖI: Quá thời gian thực thi (30s)"
        return {**state, "intermediate_answer": intermediate}
    except Exception as e:
        intermediate["code_output"] = f"LỖI: {e}"
        return {**state, "intermediate_answer": intermediate}


def logic_explanation_node(state: AgentState) -> AgentState:
    """Node 3: Formatting & Explanation (Định dạng & Giải thích).
    
    Nhận kết quả từ Solver và dùng LLM để viết lại thành lời giải dễ hiểu.
    Nếu Solver thất bại, node này sẽ ưu tiên dùng kết quả từ nhánh Direct (fallback).

    Args:
        state: Trạng thái chứa kết quả solver và fallback.

    Returns:
        Trạng thái cập nhật đáp án cuối cùng (final_answer).
    """
    try:
        from src.llm.factory import LLMFactory
        llm = LLMFactory.create_client(purpose="summary").get_llm()

        intermediate = state.get("intermediate_answer", {})
        code_output = intermediate.get("code_output", "")
        fallback = state.get("fallback_answer", {})

        # Kiểm tra xem Z3 có trả về đáp án hợp lệ không
        z3_success = False
        if "ANSWER:" in code_output:
            z3_success = True
            logger.info("Z3 trả về kết quả hợp lệ, đang tạo giải thích.")
        else:
            logger.warning("Z3 thất bại, chuyển sang sử dụng kết quả dự phòng từ LLM.")

        if z3_success:
            prompt = LOGIC_OUTPUT_PROMPT.format(
                question=state['question'],
                code_output=code_output
            )
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content=prompt)])
            content = response.content
            answer, reasoning = _parse_output(content, code_output)
        else:
            # Sử dụng kết quả suy luận trực tiếp từ logic_direct_node
            answer = fallback.get("answer", "Không xác định")
            reasoning = fallback.get("reasoning", "Bộ giải ký hiệu thất bại và không có giải trình thay thế.")
            logger.info("Sử dụng đáp án dự phòng.")
        
        final = {
            "answer": answer,
            "reasoning": reasoning,
            "final_output": f"Đáp án:\n{answer}\n\nLập luận:\n{reasoning}"
        }
        return {"final_answer": final}

    except Exception as e:
        logger.error(f"Tạo giải thích logic thất bại: {e}")
        intermediate = state.get("intermediate_answer", {})
        code_out = intermediate.get("code_output", "")
        answer = _extract_answer_from_code_output(code_out)
        final = {
            "answer": answer,
            "reasoning": "Kết quả được trích xuất trực tiếp từ bộ giải Z3.",
            "final_output": f"Đáp án:\n{answer}\n\nLập luận:\nKết quả được trích xuất trực tiếp từ bộ giải Z3."
        }
        return {"final_answer": final}


def logic_direct_node(state: AgentState) -> AgentState:
    """Parallel Node: Direct LLM Reasoning (Suy luận trực tiếp).
    
    Cung cấp đáp án trực tiếp từ LLM dựa trên khả năng suy luận nội tại.
    Node này chạy song song để làm "lưới an toàn" nếu nhánh Z3 gặp lỗi.

    Args:
        state: Trạng thái chứa câu hỏi.

    Returns:
        Trạng thái cập nhật kết quả dự phòng vào `fallback_answer`.
    """
    try:
        from src.llm.factory import LLMFactory
        llm = LLMFactory.create_client(purpose="reasoning").get_llm()

        premises_text = "\n".join([f"- {p}" for p in state.get("premises", [])])
        context_block = f"Premises:\n{premises_text}\n\n" if premises_text else ""

        prompt = LOGIC_DIRECT_PROMPT.format(question=f"{context_block}{state['question']}")
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=prompt)])
        
        answer, reasoning = _parse_direct_output(response.content)
        logger.info("Đã tạo đáp án suy luận trực tiếp (nhánh song song).")
        
        fallback = {
            "answer": answer,
            "reasoning": reasoning,
            "final_output": f"Đáp án:\n{answer}\n\nLập luận:\n{reasoning}"
        }
        return {"fallback_answer": fallback}
    except Exception as e:
        logger.error(f"Nhánh suy luận trực tiếp thất bại: {e}")
        return state


def _extract_code(text: str) -> str:
    """Extracts a Python code block from the LLM's raw text response.

    Args:
        text: The raw string response from the LLM.

    Returns:
        The content of the first Python code block found, or the original text 
        if no code block markers are present.
    """
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_answer_from_code_output(output: str) -> str:
    """Extracts the specific answer label from the solver's stdout.

    Args:
        output: The string output from executing the Z3 code.

    Returns:
        The extracted answer string (e.g., 'A', 'B') or 'Unknown' if not found.
    """
    match = re.search(r"ANSWER:\s*(.+)", output, re.IGNORECASE)
    return match.group(1).strip() if match else "Unknown"


def _parse_output(content: str, fallback: str) -> tuple[str, str]:
    """Parses structured Answer and Reasoning from the LLM's explanation output.

    Args:
        content: The text response from the explanation LLM.
        fallback: The solver's raw output to use if parsing fails.

    Returns:
        A tuple containing (answer, reasoning).
    """
    answer = "Unknown"
    reasoning = content

    answer_match = re.search(r"Answer:\s*\n?(.*?)(?:\n\nReasoning:|\Z)", content, re.DOTALL)
    reasoning_match = re.search(r"Reasoning:\s*\n?(.*)", content, re.DOTALL)

    if answer_match:
        answer = answer_match.group(1).strip()
    elif fallback:
        answer = _extract_answer_from_code_output(fallback)

    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    return answer, reasoning


def _parse_direct_output(content: str) -> tuple[str, str]:
    """Parses the LLM's direct reasoning response.

    Args:
        content: The raw text response from the LLM reasoning path.

    Returns:
        A tuple containing (answer, reasoning).
    """
    reasoning = ""
    answer = ""
    
    reasoning_match = re.search(r"Reasoning:\s*\n?(.*)", content, re.DOTALL)
    answer_match = re.search(r"Answer:\s*\n?(.*?)(?:\n\nReasoning:|\Z)", content, re.DOTALL)
    
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    if answer_match:
        answer = answer_match.group(1).strip()
        
    return answer, reasoning
