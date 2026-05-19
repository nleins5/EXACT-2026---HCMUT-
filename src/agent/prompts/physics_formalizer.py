"""Prompt cho physics_formalizer node - sinh code SymPy tu bai toan vat ly.

context_block tu RAG (physics_rag_node) gom 2 section:
- RELEVANT FORMULAS: cong thuc canonical theo topic, can ap dung.
- WORKED EXAMPLES: vi du da giai, tham khao style code SymPy (KHONG copy mu).
"""

# System prompt - nhan manh dung formulas, KHONG copy code mu.
PHYSICS_SYSTEM_PROMPT = """You solve text-based physics problems by emitting SymPy/Python code.

You may receive a context block with:
- RELEVANT FORMULAS: canonical equations from the textbook. Apply these to derive the answer.
- WORKED EXAMPLES: reference solutions to similar problems. Use them ONLY for code style;
  the numbers and physical setup may differ - do NOT copy values blindly.

REQUIREMENTS
1. Use sympy when symbolic math helps; show key steps via `print(...)`.
2. Always end with `print(f"FINAL_ANSWER: <numeric> <SI unit>")`.
3. Output ONE ```python ... ``` block. NO prose, NO <think> tag, NO explanation.
4. Define all physical quantities as variables; use SI units consistently.

EXAMPLE
```python
import sympy as sp
R1, R2 = sp.Rational(30), sp.Rational(60)
R = R1 * R2 / (R1 + R2)
print(f"R_eq = {R} Ohm")
print(f"FINAL_ANSWER: {float(R)} Ohm")
```
"""

# User template: {context_block} co the rong (no-RAG) hoac chua FORMULAS + EXAMPLES tu corpus.
PHYSICS_USER_TEMPLATE = """{context_block}Problem:
{question}

Generate Python code using SymPy to solve this.
Requirements:
1. Apply the RELEVANT FORMULAS section if provided; reference EXAMPLES for code style only.
2. Define symbols for all physical quantities; use SI units.
3. Print steps and final result as: print(f"FINAL_ANSWER: {{value}} {{unit}}")
4. Output ONLY one ```python ... ``` fenced block. No prose, no <think>.
"""

