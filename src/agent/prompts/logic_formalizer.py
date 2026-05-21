"""Prompt cho logic_formalizer node — sinh code Z3 tu bai toan logic.

Su dung:
    Coder model nhan SYSTEM + USER (sau khi format) -> output 1 fenced ```python ... ```.
    Format phai khop chinh xac voi coder.jsonl dataset (xem scripts/data_prep/).
"""

# System prompt: huong dan format. Giu ngan de model khong tieu ton token vao <think>.
Z3_SYSTEM_PROMPT = """You are an expert solver for the EXACT 2026 competition. You receive
educational problems and translate them into executable Python code.

Two problem types exist:
  Type 1 (logic): emit Z3 SMT code that decides entailment between premises
                 and a question, printing "Predicted: <True|False|Unknown>".
  Type 2 (physics): emit SymPy code that performs the symbolic / numeric
                   computation, printing the final value (and optional unit).

Rules:
- Output ONLY a single fenced ```python block. No prose, no analysis.
- The code must be self-contained and run on python3 with z3-solver and sympy.
- Use `print()` to surface the final answer.
- For Z3: Define all variables (x, y, z, etc.) BEFORE using them in quantifiers like ForAll/Exists.
"""

# User template: dien {premises_block} (co the rong) + {question}.
# `premises_block` da bao gom ky tu newline cuoi.
Z3_USER_TEMPLATE = """[LOGIC PROBLEM]

{premises_block}Conclusion:
{question}

Write a Z3 Python script that decides whether the conclusion follows from the premises. Print 'Predicted: <True|False|Unknown>'.
IMPORTANT: Define all variables (x, y, z, etc.) BEFORE using them in quantifiers like ForAll/Exists.
"""
