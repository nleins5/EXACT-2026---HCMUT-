"""Build coder.jsonl for fine-tuning Qwen2.5-Coder-7B-Instruct.

The Coder model handles **code generation only**:
  - Type 1 (Logic) input  -> Z3 Python script
  - Type 2 (Physics) input -> SymPy Python script

Output schema (each line is one ChatML record):
{
  "messages": [
    {"role": "system",    "content": "..."},
    {"role": "user",      "content": "<problem statement>"},
    {"role": "assistant", "content": "```python\\n<code>\\n```"}
  ],
  "meta": {"source": "btc_logic|btc_physics|electro", "type": "logic|physics", "uid": "..."}
}

Sources:
  1. BTC Logic (411 records / 808 Q-A) -> NL+FOL premises -> Z3 code
  2. BTC Physics (1,352 rows after Q19 filter) -> question + cot -> SymPy template
  3. electro_sympy_dataset.jsonl (~217 verified) -> question -> SymPy code

Only records whose generated code passes `exec()` are kept (configurable).
A 90/10 train/val split is written; STATS.md summarises the corpus.

Usage:
  python -m scripts.data_prep.prepare_coder_dataset
  python -m scripts.data_prep.prepare_coder_dataset --no-verify --val-ratio 0.05
  python -m scripts.data_prep.prepare_coder_dataset --output-dir data/finetune
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from collections import Counter
from pathlib import Path

# Make sibling import work when run as script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_prep._common import (  # noqa: E402
    OUTPUT_DIR,
    SEED,
    VAL_RATIO,
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

SYS_CODER = textwrap.dedent("""
    You are an expert solver for the EXACT 2026 competition. You receive
    educational problems and translate them into executable Python code.

    Two problem types exist:
      Type 1 (logic): emit Z3 SMT code that decides entailment between premises
                     and a question, printing "Predicted: <Yes|No|Unknown>".
      Type 2 (physics): emit SymPy code that performs the symbolic / numeric
                       computation, printing the final value (and optional unit).

    Rules:
    - Output ONLY a single fenced ```python block. No prose, no analysis.
    - The code must be self-contained and run on python3 with z3-solver and sympy.
    - Use `print()` to surface the final answer.
""").strip()


# ═══════════════════════════════════════════════════════════════════════════
# Builders for each source
# ═══════════════════════════════════════════════════════════════════════════

def _wrap_python(code: str) -> str:
    return f"```python\n{code.strip()}\n```"


def build_logic_records(verify: bool) -> list[dict]:
    """FOLIO -> Z3 code records.

    Note: BTC Logic has no conclusion-FOL field, so we cannot synthesise solid
    Z3 entailment scripts from it. We use FOLIO instead, which has full
    premises-FOL + conclusion-FOL + label.
    """
    print("[1/3] Building FOLIO -> Z3 records...")
    items = load_folio()
    print(f"      loaded {len(items)} FOLIO records")

    records: list[dict] = []
    n_skip_exec = 0
    n_skip_empty = 0

    for s in items:
        try:
            code = folio_to_z3(s)
        except Exception:
            n_skip_empty += 1
            continue
        if not code or "Could not auto-convert" in code:
            n_skip_empty += 1
            continue

        if verify:
            res = verify_python(code)
            if not res.ok:
                n_skip_exec += 1
                continue

        user_msg = (
            "[LOGIC PROBLEM]\n\n"
            f"Premises:\n{format_premises_block(s.premises_nl, s.premises_fol)}\n\n"
            f"Conclusion:\n{s.conclusion_nl}\n\n"
            f"Conclusion (FOL): {s.conclusion_fol}\n\n"
            f"Expected label: {s.label}\n\n"
            "Write a Z3 Python script that decides whether the conclusion follows "
            "from the premises. Print 'Predicted: <True|False|Unknown>'."
        )
        records.append(
            chatml(
                SYS_CODER,
                truncate(user_msg, 6000),
                _wrap_python(code),
                meta={"source": "folio", "type": "logic", "uid": s.uid},
            )
        )

    print(f"      kept {len(records)} | skipped exec={n_skip_exec} empty={n_skip_empty}")
    return records


def build_physics_btc_records(verify: bool) -> list[dict]:
    """BTC Physics -> SymPy template records.

    The BTC CSV gives us (question, cot, answer, unit). We convert the answer
    to a SymPy verification stub via the legacy engine. Where conversion fails,
    we emit a numeric-only stub that prints the answer (still trains the model
    to produce a runnable program with the right output).
    """
    print("[2/3] Building BTC Physics -> SymPy records...")
    items = load_btc_physics()
    print(f"      loaded {len(items)} BTC Physics rows (Q19-filtered)")

    sympy_eng = get_sympy_engine()
    records: list[dict] = []
    n_skip_exec = 0

    for q in items:
        # Synthesise a minimal record dict the legacy engine can consume.
        ans_token = q.answer
        latex_answer = f"= {ans_token}"
        synthetic = {"id": q.id, "final_answers": [latex_answer]}
        try:
            code = sympy_eng.generate_sympy_code(synthetic)
        except Exception:
            code = None

        if not code:
            # Numeric-only fallback so the assistant message is still valid Python
            code = (
                f'"""Physics problem {q.id} - numeric stub"""\n'
                f"from sympy import *\n"
                f"answer = {q.answer!r}\n"
                f"unit = {q.unit!r}\n"
                f'print(f"Final answer: {{answer}} {{unit}}".strip())'
            )

        if verify:
            res = verify_python(code)
            if not res.ok:
                n_skip_exec += 1
                continue

        user_msg = (
            "[PHYSICS PROBLEM]\n\n"
            f"{q.question}\n\n"
        )
        if q.cot:
            user_msg += f"Reasoning hint:\n{q.cot}\n\n"
        user_msg += (
            f"Expected final answer: {q.final_answer}\n\n"
            "Write a SymPy Python script that performs the calculation and "
            "prints the final answer."
        )

        records.append(
            chatml(
                SYS_CODER,
                truncate(user_msg, 6000),
                _wrap_python(code),
                meta={"source": "btc_physics", "type": "physics", "uid": f"btc_{q.id}"},
            )
        )

    print(f"      kept {len(records)} | skipped exec={n_skip_exec}")
    return records


def build_electro_records(verify: bool) -> list[dict]:
    """electro_sympy_dataset.jsonl -> SymPy code records.

    These are textbook physics problems that already have verified executable
    SymPy code attached.
    """
    print("[3/3] Building electro_sympy -> SymPy records...")
    items = load_electro_sympy()
    print(f"      loaded {len(items)} electro samples with SymPy code")

    records: list[dict] = []
    n_skip_exec = 0

    for s in items:
        if verify:
            res = verify_python(s.sympy_code)
            if not res.ok:
                n_skip_exec += 1
                continue

        user_msg = (
            "[PHYSICS PROBLEM]\n\n"
            f"{truncate(s.question, 2500)}\n\n"
        )
        if s.final_answers:
            user_msg += "Expected final answer(s):\n"
            for fa in s.final_answers[:6]:
                user_msg += f"  - {truncate(fa, 250)}\n"
            user_msg += "\n"
        user_msg += (
            "Write a SymPy Python script that performs the calculation and "
            "prints the final answer."
        )

        records.append(
            chatml(
                SYS_CODER,
                truncate(user_msg, 6000),
                _wrap_python(s.sympy_code),
                meta={"source": "electro", "type": "physics", "uid": s.id},
            )
        )

    print(f"      kept {len(records)} | skipped exec={n_skip_exec}")
    return records


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Prepare coder dataset for Qwen2.5-Coder-7B fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO,
                        help=f"Validation split ratio (default: {VAL_RATIO})")
    parser.add_argument("--seed", type=int, default=SEED, help=f"Shuffle seed (default: {SEED})")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip exec() verification of generated code (faster)")
    parser.add_argument("--no-electro", action="store_true",
                        help="Skip the electro_sympy source")
    args = parser.parse_args()

    verify = not args.no_verify

    print("=" * 70)
    print("EXACT 2026 - Coder Dataset Preparation")
    print("=" * 70)

    all_records: list[dict] = []
    all_records.extend(build_logic_records(verify=verify))
    all_records.extend(build_physics_btc_records(verify=verify))
    if not args.no_electro:
        all_records.extend(build_electro_records(verify=verify))

    by_source = Counter(r["meta"]["source"] for r in all_records)
    print()
    print(f"Total kept: {len(all_records)}  (by source: {dict(by_source)})")

    train, val = train_val_split(all_records, val_ratio=args.val_ratio, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "coder.jsonl"
    val_path = args.output_dir / "coder.eval.jsonl"
    stats_path = args.output_dir / "coder.STATS.md"

    print()
    write_jsonl(train, train_path)
    write_jsonl(val, val_path)
    write_stats_md(all_records, stats_path, title="Coder dataset (Qwen2.5-Coder-7B)")

    print()
    print("=" * 70)
    print(f"Done. Train={len(train)}  Val={len(val)}")
    print(f"      {train_path}")
    print(f"      {val_path}")
    print(f"      {stats_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
