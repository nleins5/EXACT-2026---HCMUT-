"""Logic Solver Node — execute Z3 code in a sandboxed subprocess."""
import subprocess
import sys
from src.agent.state import AgentState
from src.core.config import settings
from src.utils.logger import logger
from src.utils.z3_output_parser import parse_z3_output

_SOLVER_TIMEOUT = settings.solver.timeout_s


def logic_solver_node(state: AgentState) -> dict:
    intermediate = state.get("intermediate_answer", {})
    code = intermediate.get("generated_code", "")

    if not code:
        intermediate["code_output"] = ""
        intermediate["code_error"] = True
        intermediate["error_message"] = "No code to execute (formalizer returned empty)."
        return {"intermediate_answer": intermediate}

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=_SOLVER_TIMEOUT,
        )
        if result.returncode == 0:
            raw_output = result.stdout.strip()

            prediction = parse_z3_output(raw_output)

            intermediate["code_output"] = f"Predicted: {prediction}"
            intermediate["code_error"] = False
            intermediate["error_message"] = ""
            logger.info(f"Z3 Solver OK: {raw_output[:200]} -> Normalized: Predicted: {prediction}")
        else:
            intermediate["code_output"] = result.stdout.strip()
            intermediate["code_error"] = True
            intermediate["error_message"] = result.stderr.strip() or "Unknown runtime error"
            logger.warning(f"Z3 Solver lỗi: {intermediate['error_message'][:200]}")

    except subprocess.TimeoutExpired:
        intermediate["code_output"] = ""
        intermediate["code_error"] = True
        intermediate["error_message"] = f"Execution timed out ({_SOLVER_TIMEOUT}s)."
        logger.warning(f"Z3 Solver: timeout {_SOLVER_TIMEOUT}s")
    except Exception as e:
        intermediate["code_output"] = ""
        intermediate["code_error"] = True
        intermediate["error_message"] = f"Subprocess error: {e}"
        logger.error(f"Z3 Solver exception: {e}")

    return {"intermediate_answer": intermediate}
