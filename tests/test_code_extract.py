"""Unit tests for code_extract — all 4 extraction strategies + edge cases.

The code extractor determines whether LLM output becomes executable code
or an empty string (triggering error-branch). Bugs here cause silent
mis-routing in the pipeline.
"""
import pytest
from src.utils.code_extract import extract_python_code


# ── Strategy 1: Strip <think> blocks ───────────────────────────────

class TestThinkBlockStripping:
    def test_think_block_removed(self):
        text = "<think>Let me analyze this...</think>\n```python\nprint('hello')\n```"
        assert extract_python_code(text) == "print('hello')"

    def test_think_block_multiline(self):
        text = "<think>\nStep 1: ...\nStep 2: ...\n</think>\n```python\nfrom z3 import *\nprint('done')\n```"
        result = extract_python_code(text)
        assert "from z3 import *" in result
        assert "print('done')" in result

    def test_think_block_case_insensitive(self):
        text = "<THINK>reasoning</THINK>\n```python\nprint('ok')\n```"
        assert extract_python_code(text) == "print('ok')"

    def test_think_block_without_code(self):
        text = "<think>Just some reasoning without any code</think>\nNo code here"
        assert extract_python_code(text) == ""


# ── Strategy 2: Closed fenced block ```python ... ``` ──────────────

class TestClosedFence:
    def test_python_fence(self):
        text = "Here is the code:\n```python\nfrom z3 import *\nprint('hello')\n```\nDone."
        result = extract_python_code(text)
        assert "from z3 import *" in result
        assert "print('hello')" in result

    def test_py_fence(self):
        text = "```py\nprint('test')\n```"
        assert extract_python_code(text) == "print('test')"

    def test_bare_fence(self):
        text = "```\nimport sympy\nprint(42)\n```"
        result = extract_python_code(text)
        assert "import sympy" in result

    def test_multiple_fences_takes_first(self):
        text = "```python\nprint('first')\n```\n\n```python\nprint('second')\n```"
        assert extract_python_code(text) == "print('first')"

    def test_fence_with_trailing_spaces(self):
        text = "```python   \nprint('ok')\n```"
        assert extract_python_code(text) == "print('ok')"

    def test_multiline_z3_code(self):
        text = """Here is the Z3 solution:
```python
from z3 import *
Object = DeclareSort('Object')
Dog = Function('Dog', Object, BoolSort())
buddy = Const('buddy', Object)
s = Solver()
x = Const('x', Object)
s.add(ForAll([x], Implies(Dog(x), Dog(x))))
s.add(Dog(buddy))
print("Predicted: True")
```
"""
        result = extract_python_code(text)
        assert "from z3 import *" in result
        assert "Predicted: True" in result
        assert "```" not in result


# ── Strategy 3: Open fence (truncated by max_tokens) ───────────────

class TestOpenFence:
    def test_truncated_fence(self):
        text = "```python\nfrom z3 import *\nprint('truncated')"
        result = extract_python_code(text)
        assert "from z3 import *" in result
        assert "print('truncated')" in result

    def test_truncated_without_newline(self):
        text = "```python\nprint('no_end')"
        assert extract_python_code(text) == "print('no_end')"


# ── Strategy 4: No fence, code hint heuristic ──────────────────────

class TestHeuristic:
    def test_import_statement(self):
        text = "from z3 import *\nprint('hello')"
        result = extract_python_code(text)
        assert "from z3 import *" in result

    def test_sympy_keyword(self):
        text = "import sympy as sp\nx = sp.symbols('x')\nprint(x)"
        result = extract_python_code(text)
        assert "import sympy" in result

    def test_solver_keyword(self):
        text = "s = Solver()\ns.add(x > 0)"
        result = extract_python_code(text)
        assert "Solver()" in result

    def test_print_keyword(self):
        text = "result = 42\nprint(result)"
        result = extract_python_code(text)
        assert "print(result)" in result

    def test_no_code_hints(self):
        """Pure prose should return empty string."""
        text = "The answer is True because all birds can fly."
        assert extract_python_code(text) == ""

    def test_define_keyword(self):
        text = "def solve():\n    return 42"
        result = extract_python_code(text)
        assert "def solve():" in result


# ── Edge Cases ─────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_string(self):
        assert extract_python_code("") == ""

    def test_none_returns_empty(self):
        assert extract_python_code(None) == ""

    def test_whitespace_only(self):
        assert extract_python_code("   \n\n   ") == ""

    def test_fence_with_empty_code(self):
        text = "```python\n\n```"
        # Empty fence — should return empty string after strip
        result = extract_python_code(text)
        assert result == ""

    def test_think_then_closed_fence(self):
        """Full DeepSeek-R1 style output."""
        text = """<think>
I need to solve this logic problem using Z3.
Let me define the predicates...
</think>

```python
from z3 import *
Entity = DeclareSort('Entity')
Dog = Function('Dog', Entity, BoolSort())
x = Const('x', Entity)
buddy = Const('buddy', Entity)
s = Solver()
s.add(ForAll([x], Implies(Dog(x), Dog(x))))
s.add(Dog(buddy))
conclusion = Dog(buddy)
s.push()
s.add(Not(conclusion))
r1 = s.check()
s.pop()
if r1 == unsat:
    print("Predicted: True")
else:
    print("Predicted: Unknown")
```"""
        result = extract_python_code(text)
        assert "from z3 import *" in result
        assert "<think>" not in result
        assert "Predicted:" in result

    def test_priority_closed_over_open(self):
        """Closed fence should be preferred over open fence interpretation."""
        text = "```python\nprint('closed')\n```\n\nSome trailing text"
        assert extract_python_code(text) == "print('closed')"
