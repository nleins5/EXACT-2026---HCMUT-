"""Build instruct.jsonl for fine-tuning Qwen2.5-7B-Instruct.

The Instruct model handles **explanation generation only**. It receives:
  - the original problem
  - the premises (logic) or question (physics)
  - the Coder model's generated code
  - the runtime output (or error message) from executing that code

and produces a JSON response matching the `ExactResponse` schema in
`src/api/schemas.py`:

    {
      "answer": "...",
      "explanation": "...",
      "fol":  "..."     | null,
      "cot":  ["..."]   | null,
      "premises": ["Premise 1: ...", ...],
      "confidence": 0.85
    }

Note on `fol` / `cot` typing (mirrors EXACT_Slides.pdf p.33 + src/api/schemas.py):
  - `fol` is a single FOL formula string (e.g. `"ForAll(x, P(x) -> Q(x))"`).
  - `cot` is a list of step strings (e.g. `["Step 1: ...", "Step 2: ..."]`).
Logic problems populate `fol` and leave `cot=null`; physics problems do the opposite.

Two branches are produced to mirror the runtime two-prompt strategy in
`src/agent/nodes/*_explanation.py`:

  - SUCCESS branch: code executed cleanly, code_output is real stdout.
                    Target = ground-truth answer + structured explanation.
                    ~70% of records.
  - ERROR branch:   code raised, code_output is the error string.
                    Target = answer derived from CoT/explanation directly,
                             with a graceful explanation acknowledging the
                             code failure.
                    ~30% of records.

Usage:
  python -m scripts.data_prep.prepare_instruct_dataset
  python -m scripts.data_prep.prepare_instruct_dataset --error-ratio 0.4
  python -m scripts.data_prep.prepare_instruct_dataset --output-dir data/finetune
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_prep._common import (  # noqa: E402
    OUTPUT_DIR,
    SEED,
    VAL_RATIO,
    FolioSample,
    PhysicsQA,
    chatml,
    folio_to_z3,
    format_premises_block,
    get_sympy_engine,
    load_btc_physics,
    load_electro_sympy,
    load_folio,
    train_val_split,
    truncate,
    verify_python,
    write_jsonl,
    write_stats_md,
)


# ═══════════════════════════════════════════════════════════════════════════
# System prompt
# ═══════════════════════════════════════════════════════════════════════════

SYS_INSTRUCT = textwrap.dedent("""
    You are the explanation engine for the EXACT 2026 competition. You are
    given an educational problem along with the output of a symbolic solver
    (Z3 for logic, SymPy for physics). Your sole job is to produce a single
    JSON object that matches this schema exactly:

      {
        "answer": "<final answer string>",
        "explanation": "<concise reasoning paragraph>",
        "fol":   "<single FOL formula string>"             | null,
        "cot":   ["<step 1>", "<step 2>", ...]             | null,
        "premises": ["Premise 1: ...", "Premise 2: ...", ...],
        "confidence": <float between 0 and 1>
      }

    Rules:
    - Output ONLY the JSON object. No markdown fences. No prose outside the JSON.
    - For logic problems, populate `fol` (one formula string) and `premises`;
      set `cot` to null.
    - For physics problems, populate `cot` (a list of step strings) and
      `premises`; set `fol` to null.
    - If the solver output contains an error, derive the answer from your own
      reasoning over the problem text and explain that the solver failed.
""").strip()


# ═══════════════════════════════════════════════════════════════════════════
# ExactResponse target builder
# ═══════════════════════════════════════════════════════════════════════════

def _exact_response(
    *,
    answer: str,
    explanation: str,
    fol: str | None,
    cot: list[str] | None,
    premises: list[str],
    confidence: float,
) -> str:
    """Render an ExactResponse JSON object as a compact string.

    Schema mirrors `src/api/schemas.py::PredictResponse` (BTC slide 33):
        fol: single FOL formula string, e.g. "ForAll(x, P(x) -> Q(x))".
        cot: list of reasoning step strings, e.g. ["Step 1...", "Step 2..."].
    """
    obj = {
        "answer": answer,
        "explanation": explanation,
        "fol": fol,
        "cot": cot,
        "premises": premises,
        "confidence": round(confidence, 2),
    }
    return json.dumps(obj, ensure_ascii=False)


def _split_to_steps(text: str, *, max_steps: int = 8, max_chars: int = 400) -> list[str]:
    """Convert a free-form CoT/solution paragraph into a list of step strings.

    Splits on newlines first; if the result is a single line, fall back to
    sentence boundaries. Each step is hard-truncated to keep records bounded.
    Returns an empty list for empty input so callers can pass `None` to JSON.
    """
    if not text or not text.strip():
        return []
    # First try newline-based split (handles "Step 1:\nStep 2:\n" and bullet lists).
    raw = [s.strip("•-* \t") for s in text.split("\n") if s.strip()]
    if len(raw) < 2:
        # Fall back to sentence split on ". " (avoid splitting decimals like 3.14).
        import re as _re
        raw = [s.strip() for s in _re.split(r"(?<=[a-zA-Z\)])\.\s+", text) if s.strip()]
    steps: list[str] = []
    for s in raw[:max_steps]:
        if len(s) > max_chars:
            s = s[: max_chars - 1] + "…"
        steps.append(s)
    return steps


def _format_premises_for_response(premises_nl: list[str]) -> list[str]:
    """Convert raw NL premises into the 'Premise N: ...' format expected by API."""
    return [f"Premise {i}: {p}" for i, p in enumerate(premises_nl, 1)]


# ═══════════════════════════════════════════════════════════════════════════
# User-message builder
# ═══════════════════════════════════════════════════════════════════════════

def _build_user_msg(
    *,
    problem_type: str,
    problem_text: str,
    premises_block: str,
    code: str,
    code_output: str,
    code_error: bool,
) -> str:
    branch = "ERROR" if code_error else "SUCCESS"
    out_label = "Solver error" if code_error else "Solver stdout"
    return textwrap.dedent(f"""
        [{problem_type.upper()} PROBLEM]

        {problem_text}

        Premises / context:
        {premises_block}

        Solver code:
        ```python
        {truncate(code, 3000)}
        ```

        {out_label} ({branch} branch):
        ```
        {truncate(code_output.strip() or '(empty)', 1500)}
        ```

        Produce the ExactResponse JSON object now.
    """).strip()


# ═══════════════════════════════════════════════════════════════════════════
# Logic builders (success + error branches)
# ═══════════════════════════════════════════════════════════════════════════

def _logic_target(s: FolioSample, *, code_error: bool, code_output: str) -> str:
    """Build the ExactResponse target for a FOLIO logic problem.

    The target answer mirrors the FOLIO label but is rephrased to match the
    EXACT response style: True -> "Yes", False -> "No", Unknown -> "Unknown".
    """
    label_map = {"True": "Yes", "False": "No", "Unknown": "Unknown"}
    answer = label_map.get(s.label, s.label or "Unknown")

    if code_error:
        explanation = (
            f"The Z3 solver failed ({truncate(code_output, 200)}). "
            f"Reasoning directly: the premises {'entail' if answer == 'Yes' else 'do not entail' if answer == 'No' else 'are inconclusive about'} "
            f"the conclusion '{s.conclusion_nl}'."
        )
        confidence = 0.6
    else:
        explanation = (
            f"By systematically applying the listed premises in first-order logic, "
            f"the conclusion '{s.conclusion_nl}' is {answer}."
        )
        confidence = 0.92

    # `fol` is a single FOL formula (slide 33). For FOLIO entailment we use
    # the conclusion's FOL as the salient formula; the premise FOLs already
    # appear in the `premises` block via `format_premises_block`.
    fol_formula: str | None = s.conclusion_fol.strip() if s.conclusion_fol else None

    return _exact_response(
        answer=answer,
        explanation=truncate(explanation, 1500),
        fol=fol_formula,
        cot=None,
        premises=_format_premises_for_response(s.premises_nl),
        confidence=confidence,
    )


def build_logic_records(error_ratio: float, rng: random.Random) -> list[dict]:
    print("[1/3] Building FOLIO instruct records...")
    items = load_folio()
    print(f"      loaded {len(items)} FOLIO records")

    records: list[dict] = []
    n_no_code = 0

    for s in items:
        try:
            code = folio_to_z3(s)
        except Exception:
            n_no_code += 1
            continue
        if not code:
            n_no_code += 1
            continue

        force_error = rng.random() < error_ratio
        res = verify_python(code)
        if force_error:
            code_error = True
            code_output = res.error or "RuntimeError: solver inconclusive"
        else:
            code_error = not res.ok
            code_output = res.stdout if res.ok else (res.error or "")

        user_msg = _build_user_msg(
            problem_type="logic",
            problem_text=s.conclusion_nl,
            premises_block=format_premises_block(s.premises_nl, s.premises_fol),
            code=code,
            code_output=code_output,
            code_error=code_error,
        )
        target = _logic_target(s, code_error=code_error, code_output=code_output)

        records.append(
            chatml(
                SYS_INSTRUCT,
                truncate(user_msg, 8000),
                target,
                meta={
                    "source": "folio",
                    "type": "logic",
                    "branch": "error" if code_error else "success",
                    "uid": s.uid,
                },
            )
        )

    print(f"      kept {len(records)} | skipped no-code={n_no_code}")
    return records


# ═══════════════════════════════════════════════════════════════════════════
# Physics builders
# ═══════════════════════════════════════════════════════════════════════════

def _physics_target(q: PhysicsQA, *, code_error: bool, code_output: str) -> str:
    """Build the ExactResponse target for a BTC physics problem."""
    if code_error:
        explanation = (
            f"The SymPy solver failed ({truncate(code_output, 200)}). "
            f"Reasoning manually: {q.cot or 'apply the standard physics formula.'} "
            f"Final answer: {q.final_answer}."
        )
        confidence = 0.6
    else:
        explanation = q.cot or f"Direct application of the relevant formula yields {q.final_answer}."
        confidence = 0.9

    # Premises for physics problems: derive from the cot steps if present,
    # else use the question itself as a single premise.
    if q.cot:
        steps = [s.strip() for s in q.cot.split("\n") if s.strip()]
        premises = [f"Premise {i}: {s}" for i, s in enumerate(steps[:8], 1)]
    else:
        premises = [f"Premise 1: {q.question}"]

    # `cot` is a list of step strings (slide 33). Split the BTC CoT paragraph
    # into steps so the model learns the correct shape.
    cot_steps = _split_to_steps(q.cot) if q.cot else []

    return _exact_response(
        answer=q.final_answer,
        explanation=truncate(explanation, 1500),
        fol=None,
        cot=cot_steps or None,
        premises=premises,
        confidence=confidence,
    )


def build_physics_btc_records(error_ratio: float, rng: random.Random) -> list[dict]:
    print("[2/3] Building BTC Physics instruct records...")
    items = load_btc_physics()
    print(f"      loaded {len(items)} BTC Physics rows")

    sympy_eng = get_sympy_engine()
    records: list[dict] = []

    for q in items:
        synthetic = {"id": q.id, "final_answers": [f"= {q.answer}"]}
        try:
            code = sympy_eng.generate_sympy_code(synthetic)
        except Exception:
            code = None
        if not code:
            code = (
                f'"""Physics {q.id}"""\n'
                f"answer = {q.answer!r}\n"
                f"unit = {q.unit!r}\n"
                f'print(f"Final: {{answer}} {{unit}}".strip())'
            )

        force_error = rng.random() < error_ratio
        res = verify_python(code)
        if force_error:
            code_error = True
            code_output = res.error or "RuntimeError: SymPy could not solve"
        else:
            code_error = not res.ok
            code_output = res.stdout if res.ok else (res.error or "")

        user_msg = _build_user_msg(
            problem_type="physics",
            problem_text=q.question,
            premises_block=q.cot or "(no chain-of-thought provided)",
            code=code,
            code_output=code_output,
            code_error=code_error,
        )
        target = _physics_target(q, code_error=code_error, code_output=code_output)

        records.append(
            chatml(
                SYS_INSTRUCT,
                truncate(user_msg, 8000),
                target,
                meta={
                    "source": "btc_physics",
                    "type": "physics",
                    "branch": "error" if code_error else "success",
                    "uid": f"btc_{q.id}",
                },
            )
        )

    print(f"      kept {len(records)}")
    return records


def build_electro_records(error_ratio: float, rng: random.Random) -> list[dict]:
    print("[3/3] Building electro instruct records...")
    items = load_electro_sympy()
    print(f"      loaded {len(items)} electro samples")

    records: list[dict] = []
    for s in items:
        force_error = rng.random() < error_ratio
        res = verify_python(s.sympy_code)
        if force_error:
            code_error = True
            code_output = res.error or "RuntimeError: SymPy could not solve"
        else:
            code_error = not res.ok
            code_output = res.stdout if res.ok else (res.error or "")

        # Synthesise the answer text from final_answers (LaTeX strings)
        final = s.final_answers[0] if s.final_answers else "see solution"
        if code_error:
            explanation = (
                f"The solver failed; reasoning from the textbook solution: "
                f"{truncate(s.solution, 800)} -> {final}."
            )
            confidence = 0.55
        else:
            explanation = (
                f"Following the textbook derivation: {truncate(s.solution, 800)} -> {final}."
            )
            confidence = 0.85

        # Premises = first 6 non-empty paragraphs of the solution
        paras = [p.strip() for p in s.solution.split("\n\n") if p.strip()]
        premises = [f"Premise {i}: {truncate(p, 400)}" for i, p in enumerate(paras[:6], 1)] or [
            f"Premise 1: {truncate(s.question, 400)}"
        ]

        # Split the textbook solution into ordered step strings for `cot`.
        cot_steps = _split_to_steps(s.solution)

        target = _exact_response(
            answer=truncate(final, 200),
            explanation=truncate(explanation, 1500),
            fol=None,
            cot=cot_steps or None,
            premises=premises,
            confidence=confidence,
        )

        user_msg = _build_user_msg(
            problem_type="physics",
            problem_text=truncate(s.question, 2500),
            premises_block=truncate(s.solution, 1500),
            code=s.sympy_code,
            code_output=code_output,
            code_error=code_error,
        )

        records.append(
            chatml(
                SYS_INSTRUCT,
                truncate(user_msg, 8000),
                target,
                meta={
                    "source": "electro",
                    "type": "physics",
                    "branch": "error" if code_error else "success",
                    "uid": s.id,
                },
            )
        )

    print(f"      kept {len(records)}")
    return records


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Prepare instruct dataset for Qwen2.5-7B-Instruct fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO,
                        help=f"Validation split ratio (default: {VAL_RATIO})")
    parser.add_argument("--seed", type=int, default=SEED, help=f"Shuffle seed (default: {SEED})")
    parser.add_argument("--error-ratio", type=float, default=0.30,
                        help="Probability of routing a sample through the ERROR branch (default 0.30)")
    parser.add_argument("--no-electro", action="store_true",
                        help="Skip the electro source")
    args = parser.parse_args()

    if not 0 <= args.error_ratio <= 1:
        parser.error("--error-ratio must be in [0,1]")

    rng = random.Random(args.seed)

    print("=" * 70)
    print("EXACT 2026 - Instruct Dataset Preparation")
    print(f"Error-branch ratio: {args.error_ratio:.0%}")
    print("=" * 70)

    all_records: list[dict] = []
    all_records.extend(build_logic_records(args.error_ratio, rng))
    all_records.extend(build_physics_btc_records(args.error_ratio, rng))
    if not args.no_electro:
        all_records.extend(build_electro_records(args.error_ratio, rng))

    by_source = Counter(r["meta"]["source"] for r in all_records)
    by_branch = Counter(r["meta"]["branch"] for r in all_records)
    print()
    print(f"Total kept: {len(all_records)}")
    print(f"  by source: {dict(by_source)}")
    print(f"  by branch: {dict(by_branch)}")

    train, val = train_val_split(all_records, val_ratio=args.val_ratio, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "instruct.jsonl"
    val_path = args.output_dir / "instruct.eval.jsonl"
    stats_path = args.output_dir / "instruct.STATS.md"

    print()
    write_jsonl(train, train_path)
    write_jsonl(val, val_path)
    write_stats_md(all_records, stats_path, title="Instruct dataset (Qwen2.5-7B-Instruct)")

    print()
    print("=" * 70)
    print(f"Done. Train={len(train)}  Val={len(val)}")
    print(f"      {train_path}")
    print(f"      {val_path}")
    print(f"      {stats_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
