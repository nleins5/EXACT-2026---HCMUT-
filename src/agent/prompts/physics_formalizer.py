"""Prompt cho physics_formalizer node — sinh code SymPy tu bai toan vat ly.

Khac voi logic: physics_formalizer co them context_block tu RAG node
(neu co), chua cong thuc hoac vi du SymPy lien quan.
"""

# System: nhan manh SI unit + format `FINAL_ANSWER: ...` cho solver detect.
PHYSICS_SYSTEM_PROMPT = """You solve text-based physics problems by emitting SymPy/Python code.

REQUIREMENTS
1. Use sympy when symbolic math helps; show key steps via `print(...)`.
2. Always end with `print(f"FINAL_ANSWER: <numeric> <SI unit>")`.
3. Output ONE ```python ... ``` block. NO prose, NO <think> tag, NO explanation.

EXAMPLE
```python
import sympy as sp
R1, R2 = sp.Rational(30), sp.Rational(60)
R = R1 * R2 / (R1 + R2)
print(f"R_eq = {R} Ohm")
print(f"FINAL_ANSWER: {float(R)} Ohm")
```
"""

# User template: {context_block} co the rong (no-RAG) hoac chua few-shot tu corpus.
PHYSICS_USER_TEMPLATE = """{context_block}Problem:
{question}

Generate Python code using SymPy to solve this.
Requirements:
1. Define symbols for all physical quantities; use SI units.
2. Print steps and final result as: print(f"FINAL_ANSWER: {{value}} {{unit}}")
3. Output ONLY one ```python ... ``` fenced block. No prose, no <think>.
"""
