"""
Build Final SFT Dataset — Full Pipeline (one command).

Runs the ENTIRE data pipeline end-to-end:
  Step 1: Generate SymPy code from electro_dataset.jsonl  (physics augmentation)
  Step 2: Generate Z3 code from FOLIO dataset             (logic augmentation)
  Step 3: Load BTC official data (Logic JSON + Physics CSV)
  Step 4: Merge all sources, filter executable, shuffle
  Step 5: Split train/val and export to data/colab_ready/

Usage:
  python scripts/build_final_dataset.py            # full pipeline
  python scripts/build_final_dataset.py --no-sympy  # skip SymPy generation
  python scripts/build_final_dataset.py --no-z3     # skip Z3 generation
  python scripts/build_final_dataset.py --btc-only   # BTC data only, no augmentation
"""

import csv
import json
import random
import os
from pathlib import Path
from collections import Counter
import argparse

# ============================================================
# Config
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SYSTEM_PROMPT = (
    "You are an expert educational AI assistant for the EXACT 2026 competition. "
    "For logic problems: analyze premises carefully, apply formal reasoning, "
    "and derive the correct conclusion. "
    "For physics problems: identify relevant formulas, show step-by-step calculations, "
    "and provide the final numerical answer with correct units. "
    "Always think step-by-step inside <think>...</think> tags, "
    "then give your final answer inside <answer>...</answer> tags."
)

SEED = 3407
VAL_RATIO = 0.1

# Paths
BTC_LOGIC_INPUT = (
    PROJECT_ROOT / "data" / "EXACT2026_dataset_2026-05-15"
    / "Logic_Based_Educational_Queries_Text_Only"
    / "Logic_Based_Educational_Queries.json"
)
BTC_PHYSICS_INPUT = (
    PROJECT_ROOT / "data" / "EXACT2026_dataset_2026-05-15"
    / "Physics_Problems_Text_Only"
    / "Physics_Problems_Text_Only.csv"
)
ELECTRO_INPUT = PROJECT_ROOT / "data" / "collected" / "electro_dataset.jsonl"
ELECTRO_SYMPY_OUTPUT = PROJECT_ROOT / "data" / "collected" / "electro_sympy_dataset.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "colab_ready"


# ============================================================
# Step 1: Generate SymPy augmented data (inline)
# ============================================================

def step_generate_sympy() -> list[dict]:
    """Generate SymPy code from electro_dataset, verify, return Alpaca records."""
    # Import the conversion engine from sibling script
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "convert_physics_to_sympy",
        PROJECT_ROOT / "scripts" / "convert_physics_to_sympy.py"
    )
    sympy_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sympy_mod)

    if not ELECTRO_INPUT.exists():
        print("  [WARN] electro_dataset.jsonl not found, skipping SymPy generation")
        return []

    with open(ELECTRO_INPUT, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total = len(lines)
    print(f"  Generating SymPy for {total} problems...")

    records = []
    for idx, line in enumerate(lines):
        record = json.loads(line)
        try:
            code = sympy_mod.generate_sympy_code(record)
            record['sympy_verify_code'] = code
            records.append(record)
        except Exception:
            pass

    # Verify executable
    passed = []
    for rec in records:
        try:
            exec(rec['sympy_verify_code'], {'__builtins__': __builtins__})
            passed.append(rec)
        except Exception:
            pass

    print(f"  SymPy: {len(passed)}/{total} executable")

    # Convert to Alpaca format
    alpaca = []
    for rec in passed:
        questions = rec.get('questions', [])
        instruction = ' | '.join(questions) if questions else rec['id']
        alpaca.append({
            "instruction": instruction,
            "output": rec['sympy_verify_code'],
            "source": "electro_sympy",
        })

    return alpaca


# ============================================================
# Step 2: Generate Z3 augmented data (inline)
# ============================================================

def step_generate_z3() -> list[dict]:
    """Generate Z3 code from FOLIO dataset, return Alpaca records."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "convert_logic_to_z3",
        PROJECT_ROOT / "scripts" / "convert_logic_to_z3.py"
    )
    z3_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(z3_mod)

    try:
        from datasets import load_dataset
        print("  Downloading FOLIO dataset...")
        ds = load_dataset('yale-nlp/FOLIO')
    except Exception as e:
        print(f"  [WARN] Cannot load FOLIO: {e}")
        return []

    alpaca = []
    for split_name in ['train', 'validation']:
        for rec in ds[split_name]:
            premises_fol = rec.get('premises-FOL', '')
            conclusion_fol = rec.get('conclusion-FOL', '')
            premises_nl = rec.get('premises', '')
            conclusion_nl = rec.get('conclusion', '')
            label = rec.get('label', '')

            if not premises_fol or not conclusion_fol:
                continue

            z3_code = z3_mod.fol_to_z3_code(
                premises_fol, conclusion_fol,
                premises_nl, conclusion_nl, label
            )

            instruction = (
                "You are a Logic AI Engine. Convert this logical reasoning problem "
                "into an executable Python script using the Z3 theorem prover.\n\n"
                f"[Premises (Natural Language)]:\n{premises_nl}\n\n"
                f"[Premises (FOL)]:\n{premises_fol}\n\n"
                f"[Conclusion]:\n{conclusion_nl}\n\n"
                f"[Conclusion (FOL)]:\n{conclusion_fol}\n\n"
                f"[Expected Label]: {label}"
            )

            alpaca.append({
                "instruction": instruction,
                "output": z3_code,
                "source": "folio_z3",
            })

    print(f"  Z3: {len(alpaca)} records from FOLIO")
    return alpaca


# ============================================================
# Step 3: BTC Logic Converter
# ============================================================

def convert_logic() -> list[dict]:
    """Convert BTC Logic JSON to SFT conversation samples."""
    with open(BTC_LOGIC_INPUT, "r", encoding="utf-8") as f:
        records = json.load(f)

    samples = []
    skipped = 0

    for record in records:
        premises_nl = record["premises-NL"]
        premises_fol = record.get("premises-FOL", [])
        questions = record["questions"]
        answers = record["answers"]
        explanations = record.get("explanation", [])
        idx_map = record.get("idx", [])

        for q_i, question in enumerate(questions):
            answer = answers[q_i] if q_i < len(answers) else None
            if not answer:
                skipped += 1
                continue

            if q_i < len(idx_map) and idx_map[q_i]:
                relevant_indices = idx_map[q_i]
                selected_nl = []
                selected_fol = []
                for idx_1based in relevant_indices:
                    idx_0based = idx_1based - 1
                    if 0 <= idx_0based < len(premises_nl):
                        selected_nl.append(premises_nl[idx_0based])
                    if 0 <= idx_0based < len(premises_fol):
                        selected_fol.append(premises_fol[idx_0based])
            else:
                selected_nl = premises_nl
                selected_fol = premises_fol

            premises_block = "\n".join(
                [f"{i+1}. {p}" for i, p in enumerate(selected_nl)]
            )
            user_content = (
                f"[LOGIC PROBLEM]\n"
                f"Premises:\n{premises_block}\n\n"
                f"Question:\n{question}"
            )

            explanation = explanations[q_i] if q_i < len(explanations) else ""
            fol_block = ""
            if selected_fol:
                fol_lines = "\n".join(
                    [f"  P{i+1}: {f}" for i, f in enumerate(selected_fol)]
                )
                fol_block = f"\n\nFormal Logic (FOL):\n{fol_lines}\n"

            think_content = explanation + fol_block
            assistant_content = (
                f"<think>\n{think_content.strip()}\n</think>\n"
                f"<answer>\n{answer.strip()}\n</answer>"
            )

            samples.append({
                "conversations": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ],
                "type": "logic",
            })

    print(f"[Logic] {len(samples)} samples from {len(records)} records (skipped {skipped})")
    return samples


# ============================================================
# Step 3b: BTC Physics Converter
# ============================================================

def convert_physics() -> list[dict]:
    """Convert BTC Physics CSV to SFT conversation samples."""
    samples = []
    skipped = 0

    with open(BTC_PHYSICS_INPUT, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row.get("question", "").strip()
            cot = row.get("cot", "").strip()
            answer = row.get("answer", "").strip()
            unit = row.get("unit", "").strip()

            if not question or not answer:
                skipped += 1
                continue

            user_content = f"[PHYSICS PROBLEM]\n{question}"

            if cot:
                think_content = cot
            else:
                think_content = f"Let me solve this problem step by step.\nThe answer is {answer} {unit}."

            final_answer = f"{answer} {unit}".strip() if unit else answer
            assistant_content = (
                f"<think>\n{think_content}\n</think>\n"
                f"<answer>\n{final_answer}\n</answer>"
            )

            samples.append({
                "conversations": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ],
                "type": "physics",
            })

    print(f"[Physics] {len(samples)} samples (skipped {skipped})")
    return samples


# ============================================================
# Step 4: Convert augmented Alpaca -> conversations format
# ============================================================

def augmented_to_conversations(alpaca_records: list[dict]) -> list[dict]:
    """Convert Alpaca-format augmented records to conversation format.

    Also filters out non-executable code.
    """
    samples = []
    skipped = 0

    for rec in alpaca_records:
        source = rec.get("source", "")
        instruction = rec.get("instruction", "")
        code = rec.get("output", "")

        if not instruction or not code:
            continue

        # Verify code executes
        try:
            exec(code, {'__builtins__': __builtins__})
        except Exception:
            skipped += 1
            continue

        if source == "electro_sympy":
            tag = "[PHYSICS PROBLEM]"
            sample_type = "physics_sympy"
            think_prefix = "Let me solve this physics problem using SymPy."
        elif source == "folio_z3":
            tag = "[LOGIC PROBLEM]"
            sample_type = "logic_z3"
            think_prefix = "Let me solve this logic problem using Z3."
        else:
            continue

        user_content = f"{tag}\n{instruction}"
        assistant_content = (
            f"<think>\n{think_prefix}\n</think>\n"
            f"<answer>\n```python\n{code}\n```\n</answer>"
        )

        samples.append({
            "conversations": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "type": sample_type,
        })

    return samples, skipped


# ============================================================
# Step 5: Build & Export
# ============================================================

def _write_jsonl(samples: list[dict], path: Path):
    """Write samples to JSONL file (conversations only)."""
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            line = json.dumps(
                {"conversations": sample["conversations"]},
                ensure_ascii=False,
            )
            f.write(line + "\n")
    print(f"  Wrote {len(samples)} samples -> {path}")


def build_pipeline(skip_sympy=False, skip_z3=False, btc_only=False):
    """Run the full pipeline."""
    print("=" * 60)
    print("EXACT 2026 — Full Data Pipeline")
    print("=" * 60)

    # ---- Step 1: SymPy augmentation ----
    sympy_alpaca = []
    if not btc_only and not skip_sympy:
        print("\n[Step 1/5] Generating SymPy augmented data...")
        sympy_alpaca = step_generate_sympy()
    else:
        print("\n[Step 1/5] SymPy generation skipped")

    # ---- Step 2: Z3 augmentation ----
    z3_alpaca = []
    if not btc_only and not skip_z3:
        print("\n[Step 2/5] Generating Z3 augmented data...")
        z3_alpaca = step_generate_z3()
    else:
        print("\n[Step 2/5] Z3 generation skipped")

    # ---- Step 3: BTC official data ----
    print("\n[Step 3/5] Loading BTC official data...")
    logic_samples = convert_logic()
    physics_samples = convert_physics()

    # Inject system prompt
    btc_samples = []
    for sample in logic_samples + physics_samples:
        convs = sample["conversations"]
        convs.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        btc_samples.append({
            "conversations": convs,
            "type": sample["type"],
        })

    print(f"  BTC total: {len(btc_samples)} "
          f"(logic={len(logic_samples)}, physics={len(physics_samples)})")

    # ---- Step 4: Convert augmented + filter ----
    print("\n[Step 4/5] Converting & filtering augmented data...")
    all_augmented = sympy_alpaca + z3_alpaca
    augmented_samples, aug_skipped = augmented_to_conversations(all_augmented)

    aug_types = Counter(s["type"] for s in augmented_samples)
    print(f"  Augmented: {len(augmented_samples)} passed, "
          f"{aug_skipped} filtered | {dict(aug_types)}")

    # ---- Step 5: Merge + Split + Export ----
    print("\n[Step 5/5] Merging, shuffling, and exporting...")
    all_samples = btc_samples + augmented_samples

    random.seed(SEED)
    random.shuffle(all_samples)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = len(all_samples)
    val_count = int(total * VAL_RATIO)
    train_samples = all_samples[val_count:]
    val_samples = all_samples[:val_count]

    train_path = OUTPUT_DIR / "train.jsonl"
    val_path = OUTPUT_DIR / "val.jsonl"

    _write_jsonl(train_samples, train_path)
    _write_jsonl(val_samples, val_path)

    # ---- Summary ----
    type_counts = Counter(s["type"] for s in all_samples)

    print(f"\n{'=' * 60}")
    print("Pipeline Complete!")
    print(f"{'=' * 60}")
    print(f"  Total samples:     {total}")
    for t, c in sorted(type_counts.items()):
        print(f"    |-- {t:20s} {c}")
    print(f"  Train set:         {len(train_samples)}")
    print(f"  Val set:           {len(val_samples)}")
    print(f"  Train file:        {train_path}")
    print(f"  Val file:          {val_path}")
    print(f"{'=' * 60}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='EXACT 2026 — Full Data Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                  Full pipeline (BTC + SymPy + Z3)
  %(prog)s --btc-only       BTC data only
  %(prog)s --no-sympy       Skip SymPy generation
  %(prog)s --no-z3          Skip Z3 generation
        """
    )
    parser.add_argument('--no-sympy', action='store_true',
                        help='Skip SymPy augmentation')
    parser.add_argument('--no-z3', action='store_true',
                        help='Skip Z3 augmentation')
    parser.add_argument('--btc-only', action='store_true',
                        help='Use BTC official data only (no augmentation)')

    args = parser.parse_args()
    build_pipeline(
        skip_sympy=args.no_sympy,
        skip_z3=args.no_z3,
        btc_only=args.btc_only,
    )


if __name__ == '__main__':
    main()
