"""Teacher prompts cho knowledge distillation.

Hai mode:
- EXTRACT (default): dataset BTC da co `cot` field chua reasoning + cong thuc.
  Teacher chi can TRICH XUAT formula list, symbol map, sympy code clean.
  Output ngan, deterministic, re token.
- GENERATE: dung cho problem khong co CoT, teacher tu suy luan.

Yeu cau output JSON dung schema KBRecord, khong markdown fence, khong prose ngoai JSON.
"""
from __future__ import annotations

# ────────────────────────────────────────────────────────────────────────
# EXTRACT MODE - re, dung khi dataset da co CoT
# ────────────────────────────────────────────────────────────────────────

EXTRACT_SYSTEM_PROMPT = """You are a physics formula extractor. You will receive:
- A physics problem
- A worked Chain-of-Thought (CoT) solution that ALREADY contains the correct formulas, substitutions, and final answer
- The ground-truth final answer with unit

Your job is to EXTRACT (not invent) the canonical formulas, the symbol table, and produce CLEAN runnable SymPy code that reproduces the answer.

Return ONLY a single JSON object. No markdown fences. No prose outside the JSON.

JSON SCHEMA:
{
  "topic": "<one of: electrostatics | electric_circuits | other>",
  "formulas": ["<canonical formula in plain math, e.g. 'R_eq = R1*R2/(R1+R2)'>", "..."],
  "symbols": {"<sym>": "<short description with SI unit>", ...},
  "sympy_code": "<self-contained Python that ends with print(f\\"FINAL_ANSWER: <value> <unit>\\")>",
  "answer": "<numeric answer with unit, copy from ground truth>",
  "derivation": "<1-2 sentences naming which laws were applied>"
}

EXTRACT RULES:
- "formulas": list ONLY the canonical equations cited in the CoT. Do NOT include numerical substitution lines.
- "symbols": list every symbol that appears in the formulas, with SI unit.
- "sympy_code": rewrite the CoT as runnable SymPy. Use sp.Rational for exact arithmetic. End with the FINAL_ANSWER print line.
- "topic": "electric_circuits" if the problem mentions resistors, current, voltage, Ohm, Kirchhoff, capacitors in a circuit. "electrostatics" if it mentions point charges, Coulomb, electric field, potential, capacitance of an isolated capacitor. "other" only when neither fits.
- "answer": copy the ground-truth final value + unit verbatim.

Do NOT add formulas that were not used. Do NOT change the final number.
"""

EXTRACT_FEW_SHOT_USER_1 = """Problem: A parallel circuit has R1 = 30 Ohm, R2 = 60 Ohm, U = 12 V. Calculate the equivalent resistance.

CoT:
1. Apply the parallel formula: R_eq = R1*R2/(R1+R2) = 30*60/(30+60) = 1800/90
2. Result: 20 Ohm

Final answer: 20.0 Ohm

Extract the JSON now."""

EXTRACT_FEW_SHOT_ASSISTANT_1 = """{"topic": "electric_circuits", "formulas": ["R_eq = R1*R2/(R1+R2)"], "symbols": {"R1": "first resistor (Ohm)", "R2": "second resistor (Ohm)", "R_eq": "equivalent resistance (Ohm)", "U": "supply voltage (V)"}, "sympy_code": "import sympy as sp\\nR1, R2 = sp.Rational(30), sp.Rational(60)\\nR_eq = R1*R2/(R1+R2)\\nprint(f\\"FINAL_ANSWER: {float(R_eq)} Ohm\\")", "answer": "20.0 Ohm", "derivation": "Two resistors in parallel: reciprocal-sum formula gives 20 Ohm."}"""

EXTRACT_FEW_SHOT_USER_2 = """Problem: Two point charges q1 = 2e-6 C and q2 = -3e-6 C are separated by r = 0.1 m in vacuum. Find the magnitude of the Coulomb force.

CoT:
1. Apply Coulomb's law: F = k_e * |q1*q2| / r^2
2. With k_e = 8.99e9, |q1*q2| = 6e-12, r^2 = 0.01:
   F = 8.99e9 * 6e-12 / 0.01 = 5.394 N

Final answer: 5.394 N

Extract the JSON now."""

EXTRACT_FEW_SHOT_ASSISTANT_2 = """{"topic": "electrostatics", "formulas": ["F = k_e * abs(q1*q2) / r**2"], "symbols": {"k_e": "Coulomb constant (N m^2 / C^2)", "q1": "first point charge (C)", "q2": "second point charge (C)", "r": "separation distance (m)", "F": "Coulomb force magnitude (N)"}, "sympy_code": "import sympy as sp\\nk_e = sp.Float('8.99e9')\\nq1, q2 = sp.Float('2e-6'), sp.Float('-3e-6')\\nr = sp.Float('0.1')\\nF = k_e * abs(q1*q2) / r**2\\nprint(f\\"FINAL_ANSWER: {float(F)} N\\")", "answer": "5.394 N", "derivation": "Coulomb's law applied to two point charges in vacuum."}"""


def build_extract_user_prompt(problem: str, cot: str, answer: str, unit: str = "") -> str:
    """Build user message for EXTRACT mode."""
    final = f"{answer} {unit}".strip() if unit else answer
    return (
        f"Problem: {problem.strip()}\n\n"
        f"CoT:\n{cot.strip()}\n\n"
        f"Final answer: {final}\n\n"
        f"Extract the JSON now."
    )


# ────────────────────────────────────────────────────────────────────────
# GENERATE MODE - cu, giu lai cho problem khong co CoT
# ────────────────────────────────────────────────────────────────────────

GENERATE_SYSTEM_PROMPT = """You are a senior physics teacher and Python tutor. Given a physics problem, produce a SINGLE JSON object that captures the canonical formulas, the SymPy code that solves it, and the final answer.

Return ONLY the JSON object. No markdown fences. No prose outside the JSON. No comments.

JSON SCHEMA (all fields required, use empty array/dict if not applicable):
{
  "topic": "<one of: electrostatics | electric_circuits | other>",
  "formulas": ["<canonical formula in plain math>", "..."],
  "symbols": {"<sym>": "<short description with SI unit>", ...},
  "sympy_code": "<self-contained Python that ends with print(f\\"FINAL_ANSWER: <value> <SI unit>\\")>",
  "answer": "<numeric or symbolic answer with unit>",
  "derivation": "<1-3 sentences explaining the key reasoning>"
}

RULES:
1. sympy_code must run on python3 with `sympy` installed and end with the FINAL_ANSWER line.
2. Use sp.Rational or sp.Float for precision.
3. No input(), file I/O, plotting, network.
4. Choose ONLY topic = electrostatics, electric_circuits, or other (BTC EXACT 2026 scope).
5. formulas: plain text math (use ** for power, sqrt(...) for roots), one per element.
"""

GENERATE_FEW_SHOT_USER_1 = """Problem: Two resistors R1 = 30 Ohm and R2 = 60 Ohm are connected in parallel. Find the equivalent resistance R_eq."""

GENERATE_FEW_SHOT_ASSISTANT_1 = EXTRACT_FEW_SHOT_ASSISTANT_1

GENERATE_FEW_SHOT_USER_2 = """Problem: Two point charges q1 = 2 uC and q2 = -3 uC are separated by 0.1 m in vacuum. Find the Coulomb force magnitude."""

GENERATE_FEW_SHOT_ASSISTANT_2 = EXTRACT_FEW_SHOT_ASSISTANT_2


def build_generate_user_prompt(problem: str, hint: str = "") -> str:
    parts = [f"Problem: {problem.strip()}"]
    if hint:
        parts.append(f"\nReasoning hint: {hint.strip()}")
    parts.append("\nProduce the JSON object now.")
    return "\n".join(parts)


# ────────────────────────────────────────────────────────────────────────
# Backwards-compatible aliases (cu cua teacher_client.py)
# ────────────────────────────────────────────────────────────────────────

TEACHER_SYSTEM_PROMPT = GENERATE_SYSTEM_PROMPT
FEW_SHOT_USER_1 = GENERATE_FEW_SHOT_USER_1
FEW_SHOT_ASSISTANT_1 = GENERATE_FEW_SHOT_ASSISTANT_1
FEW_SHOT_USER_2 = GENERATE_FEW_SHOT_USER_2
FEW_SHOT_ASSISTANT_2 = GENERATE_FEW_SHOT_ASSISTANT_2
build_user_prompt = build_generate_user_prompt
