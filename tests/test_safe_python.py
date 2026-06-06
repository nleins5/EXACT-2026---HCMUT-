"""Security and execution tests for the generated-code runner."""
import subprocess
import threading

import pytest

from src.utils.safe_python import (
    SolverCancelled,
    UnsafeCodeError,
    run_solver_code,
    validate_solver_code,
)


def test_rejects_non_solver_imports():
    with pytest.raises(UnsafeCodeError, match="Import is not allowed"):
        validate_solver_code("import os\nprint(os.getcwd())", allowed_imports={"z3"})


def test_rejects_private_introspection():
    with pytest.raises(UnsafeCodeError, match="Private attribute"):
        validate_solver_code("print(object.__subclasses__())", allowed_imports={"z3"})


def test_runs_allowed_z3_code():
    result = run_solver_code(
        'from z3 import *\nprint("Predicted: True")',
        allowed_imports={"z3"},
        timeout_s=8,
    )
    assert result.returncode == 0
    assert result.stdout == "Predicted: True"


def test_times_out_infinite_loop():
    with pytest.raises(subprocess.TimeoutExpired):
        run_solver_code(
            "while True:\n    pass",
            allowed_imports={"z3"},
            timeout_s=0.2,
        )


def test_rejects_excessive_output():
    with pytest.raises(UnsafeCodeError, match="output limit"):
        run_solver_code(
            'print("x" * 10000)',
            allowed_imports={"z3"},
            timeout_s=3,
            max_output_bytes=100,
        )


def test_cancel_event_stops_solver():
    event = threading.Event()
    event.set()
    with pytest.raises(SolverCancelled):
        run_solver_code(
            "while True:\n    pass",
            allowed_imports={"z3"},
            timeout_s=3,
            cancel_event=event,
        )
