"""Prompt cho logic_formalizer node — sinh code Z3 tu bai toan logic.

Su dung:
    Coder model nhan SYSTEM + USER (sau khi format) -> output 1 fenced ```python ... ```.
    Format phai khop chinh xac voi coder.jsonl dataset (xem scripts/data_prep/).
"""

# System prompt: huong dan format. Giu ngan de model khong tieu ton token vao <think>.
Z3_SYSTEM_PROMPT = """You translate natural-language logic problems into Z3 Python code.

REQUIREMENTS
1. `from z3 import *`, build a `Solver()`, add a constraint per premise.
2. Encode the question as a goal and print: `print("ANSWER: <Yes|No|Unknown|...>")`.
3. Output ONE ```python ... ``` block. NO prose, NO <think> tag, NO explanation.

EXAMPLE
Input: "Score 8 in final. Absent in lab. Regulation: 0 lab points -> cannot pass."
Output:
```python
from z3 import *
s = Solver()
score_final = Int('score_final')
absent_lab  = Bool('absent_lab')
can_pass    = Bool('can_pass')
s.add(score_final == 8)
s.add(absent_lab == True)
s.add(Implies(absent_lab, can_pass == False))
if s.check() == sat:
    m = s.model()
    print("ANSWER: No" if not m[can_pass] else "ANSWER: Yes")
else:
    print("ANSWER: Unknown")
```
"""

# User template: dien {premises_block} (co the rong) + {question}.
# `premises_block` da bao gom ky tu newline cuoi.
Z3_USER_TEMPLATE = """{premises_block}Logic Problem:
{question}

Translate the logic problem above into Python Z3 code.
Define variables for each entity and add constraints for each premise.
Output ONLY one ```python ... ``` fenced block. No prose, no <think>.
"""
