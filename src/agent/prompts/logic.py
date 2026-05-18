"""Prompt templates cho Logic Nodes (Formalizer, Explanation)."""

# Compact, instruction-first system prompt. The few-shot example is intentionally
# short — DeepSeek-R1-style models pay a heavy <think> tax on long contexts.
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

LOGIC_OUTPUT_PROMPT = """You are a Logic Explainer.

The Z3 solver ran successfully. Use its verified output to produce the final structured response.

Problem:
{question}

Z3 Solver Output (trusted):
{code_output}

Return a structured response matching the schema:
- answer: REQUIRED. (A, B, C, Yes, No, Unknown, or numeric value).
- explanation: REQUIRED. Human-readable, references the relevant premises.
- fol: optional First-Order Logic formalization.
- cot: optional list of reasoning steps.
- premises: optional list of rules used.
- confidence: optional float 0.0-1.0.

Trust the Z3 output for the final answer. No extra prose outside the schema.
"""

LOGIC_OUTPUT_ERROR_PROMPT = """You are a Logic Explainer working in fallback mode.

The Z3 solver FAILED to execute the generated code. The code still reflects the
model's earlier reasoning — read it as a structured hint, then derive the
answer yourself with natural-language inference.

Problem:
{question}

Premises:
{premises_block}

Generated Z3 code (FAILED, do NOT execute, only read for hints):
```python
{generated_code}
```

Execution error:
{error_message}

Task:
1. Read the broken Z3 code as a hint for entities/constraints the model identified.
2. Solve the problem yourself.
3. Return the structured response (answer, explanation, fol?, cot?, premises?, confidence?).
   Lower confidence since the solver failed. No prose outside the schema.
   Do NOT mention "fallback mode".
"""
