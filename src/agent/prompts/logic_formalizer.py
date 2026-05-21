"""Prompt cho logic_formalizer node — sinh code Z3 tu bai toan logic.

Su dung:
    Coder model nhan SYSTEM + USER (sau khi format) -> output 1 fenced ```python ... ```.
    Format phai khop chinh xac voi coder.jsonl dataset (xem scripts/data_prep/).

DESIGN (v2 - compatible with current fine-tuned weights):
    - Giu prompt ngan gon, gan voi format training data cu.
    - 1 one-shot example (True case) - giong training data.
    - Nhan manh pattern negate-and-check-unsat nhung KHONG thay doi format output.
    - Output parser (trong logic_solver_node) se xu ly cac format khac nhau.
"""

# System prompt: giu ngan, tuong thich voi fine-tuned model.
Z3_SYSTEM_PROMPT = """You are an expert logic solver for the EXACT 2026 competition. You translate logical reasoning problems into executable Z3 SMT Python code.

Rules:
- Output ONLY a single fenced ```python block. No prose, no analysis.
- The code must be self-contained and run on python3 with z3-solver.
- Always check entailment using the negate-and-check-unsat pattern and print the result.
- Define all variables (x, y, etc.) BEFORE using them in quantifiers like ForAll/Exists.
- Pay attention to quantifiers: "All X are Y" -> ForAll; "Some X are Y" -> Exists (does NOT mean all).

ENTAILMENT PATTERN:
1. Add premises to solver.
2. conclusion = <Z3 expression for the conclusion>
3. Negate conclusion, check unsat -> entailed (True).
4. Assert conclusion, check unsat -> contradicted (False).
5. Otherwise -> Unknown.

Here is a standard ONE-SHOT example:
---
[LOGIC PROBLEM]
Premises:
- All dogs are friendly.
- Buddy is a dog.
Conclusion:
Is Buddy friendly?

[OUTPUT CODE]
```python
from z3 import *
Object = DeclareSort('Object')
Dog = Function('Dog', Object, BoolSort())
Friendly = Function('Friendly', Object, BoolSort())
buddy = Const('buddy', Object)
s = Solver()
x = Const('x', Object)
s.add(ForAll([x], Implies(Dog(x), Friendly(x))))
s.add(Dog(buddy))
conclusion = Friendly(buddy)
s.push()
s.add(Not(conclusion))
r1 = s.check()
s.pop()
s.push()
s.add(conclusion)
r2 = s.check()
s.pop()
if r1 == unsat:
    print("Predicted: True")
elif r2 == unsat:
    print("Predicted: False")
else:
    print("Predicted: Unknown")
```
---
"""

# User template: dien {premises_block} (co the rong) + {question}.
# `premises_block` da bao gom ky tu newline cuoi.
Z3_USER_TEMPLATE = """[LOGIC PROBLEM]

{premises_block}Conclusion:
{question}

Write a Z3 Python script that decides whether the conclusion follows from the premises. Print 'Predicted: <True|False|Unknown>'.
IMPORTANT: Define all variables (x, y, z, etc.) BEFORE using them in quantifiers like ForAll/Exists.
IMPORTANT: "Some X are Y" means Exists (NOT ForAll). Only "All X are Y" means ForAll.
"""
