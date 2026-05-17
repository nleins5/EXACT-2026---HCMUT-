"""
Logic Solver Node — Thực thi mã Z3 trong subprocess an toàn.
"""
import subprocess
import sys
from src.agent.state import AgentState
from src.utils.logger import logger


def logic_solver_node(state: AgentState) -> dict:
    """Thực thi mã Z3 đã sinh ra, trả về kết quả stdout."""
    intermediate = state.get("intermediate_answer", {})
    code = intermediate.get("generated_code", "")

    if not code:
        intermediate["code_output"] = "LỖI: Không có mã code để thực thi"
        return {"intermediate_answer": intermediate}

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=30,
        )
        output = (
            result.stdout.strip()
            if result.returncode == 0
            else f"LỖI THỰC THI:\n{result.stderr}"
        )
        logger.info(f"Kết quả Solver: {output[:200]}")
        intermediate["code_output"] = output
        return {"intermediate_answer": intermediate}

    except subprocess.TimeoutExpired:
        intermediate["code_output"] = "LỖI: Quá thời gian thực thi (30s)"
        return {"intermediate_answer": intermediate}
    except Exception as e:
        intermediate["code_output"] = f"LỖI: {e}"
        return {"intermediate_answer": intermediate}
