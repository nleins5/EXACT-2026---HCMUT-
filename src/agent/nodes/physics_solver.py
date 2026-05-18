"""
Physics Solver Node — Thực thi mã SymPy trong subprocess an toàn.

Sau khi thực thi:
- Nếu thành công  → set code_error=False, lưu stdout vào code_output.
- Nếu thất bại    → set code_error=True,  lưu thông báo lỗi vào error_message.

Solver KHÔNG raise — luôn để pipeline đi tiếp đến explanation node.
"""
import subprocess
import sys
from src.agent.state import AgentState
from src.utils.logger import logger


def physics_solver_node(state: AgentState) -> dict:
    """Thực thi mã SymPy đã sinh ra, ghi nhận kết quả/lỗi vào intermediate_answer."""
    intermediate = state.get("intermediate_answer", {})
    code = intermediate.get("generated_code", "")

    if not code:
        intermediate["code_output"] = ""
        intermediate["code_error"] = True
        intermediate["error_message"] = "Không có mã code để thực thi (formalizer trả về rỗng)."
        return {"intermediate_answer": intermediate}

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            intermediate["code_output"] = result.stdout.strip()
            intermediate["code_error"] = False
            intermediate["error_message"] = ""
            logger.info(f"SymPy Solver OK: {intermediate['code_output'][:200]}")
        else:
            intermediate["code_output"] = result.stdout.strip()
            intermediate["code_error"] = True
            intermediate["error_message"] = result.stderr.strip() or "Unknown runtime error"
            logger.warning(f"SymPy Solver lỗi: {intermediate['error_message'][:200]}")

    except subprocess.TimeoutExpired:
        intermediate["code_output"] = ""
        intermediate["code_error"] = True
        intermediate["error_message"] = "Quá thời gian thực thi (30s)."
        logger.warning("SymPy Solver: timeout 30s")
    except Exception as e:
        intermediate["code_output"] = ""
        intermediate["code_error"] = True
        intermediate["error_message"] = f"Lỗi khi gọi subprocess: {e}"
        logger.error(f"SymPy Solver exception: {e}")

    return {"intermediate_answer": intermediate}
