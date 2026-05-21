"""Prompt cho logic_explanation node — sinh ExactResponse JSON.

Hai nhanh prompt mirror dataset instruct.jsonl:
- SUCCESS branch: code Z3 chay OK -> tin code_output, format JSON.
- ERROR branch: code Z3 fail -> doc code nhu hint, tu suy luan, ha confidence.
"""

# Branch khi solver thanh cong: chi can format ket qua thanh ExactResponse.
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
   
CRITICAL CONSTRAINTS:
- Do NOT mention "fallback mode".
- Do NOT mention that the Z3 solver failed or any execution errors/tracebacks in your final explanation or answer.
- Focus purely on explaining the logical deduction using the premises, as if the problem was solved directly and cleanly.
"""

