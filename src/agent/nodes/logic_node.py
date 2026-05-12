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
from src.agent.schema import ExactResponse
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

        # Nhánh logic chỉ tập trung vào việc dịch mã, không dùng RAG
        premises_text = "\n".join([f"- {p}" for p in state.get("premises", [])]) # lấy danh sách giả thiết từ state nếu có
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
            [sys.executable, "-c", code], # Chạy code python trong subprocess 
            capture_output=True, text=True, timeout=30 # Capture output và timeout 30s 
        )
        output = result.stdout.strip() if result.returncode == 0 else f"LỖI THỰC THI (RUNTIME ERROR):\n{result.stderr}" # Lấy output nếu returncode == 0 else 
        logger.info(f"Kết quả Solver: {output[:200]}") # Log output tối đa 200 ký tự
        intermediate["code_output"] = output # Lưu output vào state
        return {"intermediate_answer": intermediate}
    except subprocess.TimeoutExpired:
        intermediate["code_output"] = "LỖI: Quá thời gian thực thi (30s)" # Lưu lỗi quá thời gian
        return {**state, "intermediate_answer": intermediate}
    except Exception as e:
        intermediate["code_output"] = f"LỖI: {e}" # Lưu lỗi runtime
        return {**state, "intermediate_answer": intermediate}


def logic_explanation_node(state: AgentState) -> dict:
    """Node 3: Logic Explanation (Tổng hợp và Giải thích).
    
    Sử dụng LLM với Structured Output để tổng hợp kết quả từ Z3 Solver
    thành định dạng chuẩn của cuộc thi EXACT 2026.
    """
    intermediate = state.get("intermediate_answer", {})
    code_output = intermediate.get("code_output", "")
    
    try:
        from src.llm.factory import LLMFactory
        llm_client = LLMFactory.create_client(purpose="summary")
        structured_llm = llm_client.get_structured_llm(ExactResponse)
        
        prompt = LOGIC_OUTPUT_PROMPT.format(
            question=state["question"],
            code_output=code_output
        )
        
        # Gọi LLM và nhận trực tiếp object ExactResponse
        response: ExactResponse = structured_llm.invoke(prompt)
        
        return {"final_answer": response.model_dump()}
        
    except Exception as e:
        logger.error(f"Lỗi tại logic_explanation_node: {e}")
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


def logic_direct_node(state: AgentState) -> dict:
    """Nhánh song song: LLM suy luận trực tiếp (Dự phòng).
    
    Sử dụng LLM với Structured Output để giải bài toán trực tiếp 
    không qua bộ giải Z3.
    """
    try:
        from src.llm.factory import LLMFactory
        llm_client = LLMFactory.create_client(purpose="summary")
        structured_llm = llm_client.get_structured_llm(ExactResponse)
        
        premises_text = "\n".join([f"- {p}" for p in state.get("premises", [])])
        context_block = f"Premises:\n{premises_text}\n\n" if premises_text else ""

        prompt = LOGIC_DIRECT_PROMPT.format(question=f"{context_block}{state['question']}")
        
        # Gọi LLM và nhận object ExactResponse
        response: ExactResponse = structured_llm.invoke(prompt)
        
        return {"fallback_answer": response.model_dump()}
        
    except Exception as e:
        logger.error(f"Lỗi tại logic_direct_node: {e}")
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


def _extract_answer_from_code_output(output: str) -> str:
    """Trích xuất nhãn đáp án từ stdout của solver."""
    match = re.search(r"ANSWER:\s*(.+)", output, re.IGNORECASE)
    return match.group(1).strip() if match else "Unknown"
