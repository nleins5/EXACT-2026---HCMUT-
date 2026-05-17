"""
Convert FOLIO FOL reasoning dataset to executable Z3 verification code.

This script:
  1. Downloads the yale-nlp/FOLIO dataset from HuggingFace
  2. Converts each FOL premise + conclusion into Z3 Python code
  3. Outputs Alpaca-format JSONL for downstream merging

Usage:
  python scripts/convert_logic_to_z3.py [--output PATH]

Output format (each line):
{
  "instruction": "<problem + FOL>",
  "input": "",
  "output": "<Z3 Python code>",
  "source": "folio_z3",
  "id": "folio_<story_id>_<example_id>"
}
"""

import json
import re
import os
import argparse


# ─── Default Paths ─────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = r'd:\Exact 2026\data\sft_dataset'


# ═══════════════════════════════════════════════════════════════════════════
# FOL → Z3 Conversion Engine
# ═══════════════════════════════════════════════════════════════════════════

def fol_to_z3_code(premises_fol: str, conclusion_fol: str,
                    premises_nl: str, conclusion_nl: str,
                    label: str) -> str:
    """
    Generate Z3 Python code from FOL premises and conclusion.

    Creates a Z3 script that:
    1. Declares sorts and predicates from the FOL
    2. Encodes premises as Z3 constraints
    3. Checks if conclusion is entailed (True), contradicted (False), or Unknown
    """
    predicates = set()
    constants = set()

    all_fol = premises_fol + "\n" + conclusion_fol

    for match in re.finditer(r'([A-Z][a-zA-Z_-]*)\(', all_fol):
        predicates.add(match.group(1))

    for match in re.finditer(r'[,(]\s*([a-z][a-zA-Z0-9_]*)', all_fol):
        const = match.group(1)
        if len(const) > 1 or const not in 'xyz':
            constants.add(const)

    lines = []
    lines.append('"""')
    lines.append(f'Z3 Verification for FOLIO logical reasoning')
    lines.append(f'Premises: {premises_nl[:100]}...')
    lines.append(f'Conclusion: {conclusion_nl}')
    lines.append(f'Expected Label: {label}')
    lines.append('"""')
    lines.append('from z3 import *')
    lines.append('')

    lines.append('# Declare entity sort')
    lines.append('Entity = DeclareSort("Entity")')
    lines.append('')

    if constants:
        const_list = sorted(constants)
        lines.append('# Declare constants (entities)')
        for c in const_list:
            safe_c = re.sub(r'[^a-zA-Z0-9_]', '_', c)
            lines.append(f'{safe_c} = Const("{safe_c}", Entity)')
        lines.append('')

    if predicates:
        pred_list = sorted(predicates)
        lines.append('# Declare predicates')
        for p in pred_list:
            safe_p = re.sub(r'[^a-zA-Z0-9_]', '_', p)
            pattern = re.escape(p) + r'\(([^)]*)\)'
            m = re.search(pattern, all_fol)
            if m:
                args = [a.strip() for a in m.group(1).split(',') if a.strip()]
                arity = len(args)
            else:
                arity = 1
            arg_sorts = ', '.join(['Entity'] * arity)
            lines.append(f'{safe_p} = Function("{safe_p}", {arg_sorts}, BoolSort())')
        lines.append('')

    lines.append('# Quantifier variables')
    lines.append('x = Const("x", Entity)')
    lines.append('y = Const("y", Entity)')
    lines.append('z = Const("z", Entity)')
    lines.append('')

    lines.append('# Create solver')
    lines.append('s = Solver()')
    lines.append('')

    premises_list = premises_fol.strip().split('\n')
    lines.append('# Add premises')
    for i, prem in enumerate(premises_list, 1):
        prem = prem.strip()
        if not prem:
            continue
        lines.append(f'# Premise {i} (FOL): {prem}')
        z3_expr = _fol_line_to_z3(prem, predicates, constants)
        if z3_expr:
            lines.append(f's.add({z3_expr})  # premise_{i}')
        else:
            lines.append(f'# [SKIP] Could not auto-convert premise {i}')
    lines.append('')

    lines.append('# Check conclusion')
    lines.append(f'# Conclusion (FOL): {conclusion_fol.strip()}')
    z3_conclusion = _fol_line_to_z3(conclusion_fol.strip(), predicates, constants)
    if z3_conclusion:
        lines.append(f'conclusion = {z3_conclusion}')
    else:
        lines.append(f'# [SKIP] Could not auto-convert conclusion')
        lines.append(f'conclusion = BoolVal(True)  # placeholder')
    lines.append('')

    lines.append('# Verify entailment')
    lines.append(f'expected = "{label}"')
    lines.append('')
    lines.append('# Check if premises entail conclusion')
    lines.append('s.push()')
    lines.append('s.add(Not(conclusion))')
    lines.append('result_entails = s.check()')
    lines.append('s.pop()')
    lines.append('')
    lines.append('# Check if premises contradict conclusion')
    lines.append('s.push()')
    lines.append('s.add(conclusion)')
    lines.append('result_consistent = s.check()')
    lines.append('s.pop()')
    lines.append('')
    lines.append('if result_entails == unsat:')
    lines.append('    predicted = "True"')
    lines.append('elif result_consistent == unsat:')
    lines.append('    predicted = "False"')
    lines.append('else:')
    lines.append('    predicted = "Unknown"')
    lines.append('')
    lines.append(f'print(f"Expected: {{expected}}, Predicted: {{predicted}}")')
    lines.append(f'print(f"Match: {{expected == predicted}}")')

    return '\n'.join(lines)


def _fol_line_to_z3(fol: str, predicates: set, constants: set) -> str:
    """Convert a single FOL line to Z3 Python expression string."""
    s = fol.strip()
    if not s:
        return None

    try:
        s = re.sub(r'\u2200\s*([a-z])\s+', r'ForAll(\1, ', s)
        s = re.sub(r'\u2203\s*([a-z])\s+', r'Exists(\1, ', s)

        forall_count = s.count('ForAll(')
        exists_count = s.count('Exists(')

        s = re.sub(r'\u00ac\(', 'Not(', s)
        s = re.sub(r'\u00ac([A-Z][a-zA-Z_-]*\([^)]*\))', r'Not(\1)', s)
        s = re.sub(r'\u00ac([a-z][a-zA-Z0-9_]*)', r'Not(\1)', s)

        if '\u2295' in s:
            return None

        s = s.replace('\u2192', ', ')
        if ', ' in s and 'ForAll' not in s and 'Exists' not in s:
            parts = s.split(', ', 1)
            if len(parts) == 2:
                s = f'Implies({parts[0].strip()}, {parts[1].strip()})'

        if '\u2227' in s:
            parts = s.split('\u2227')
            if len(parts) > 1:
                inner = ', '.join(p.strip() for p in parts)
                s = f'And({inner})'

        if '\u2228' in s:
            parts = s.split('\u2228')
            if len(parts) > 1:
                inner = ', '.join(p.strip() for p in parts)
                s = f'Or({inner})'

        s = re.sub(r'([A-Z][a-zA-Z]*)-([A-Z])', r'\1_\2', s)
        s = re.sub(r'([a-z])-([A-Z])', r'\1_\2', s)

        s += ')' * (forall_count + exists_count)

        s = re.sub(r'([a-zA-Z_]+)\s*=\s*([a-zA-Z_]+)', r'\1 == \2', s)

        return s if s else None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Main: Download FOLIO + Convert
# ═══════════════════════════════════════════════════════════════════════════

def convert_folio_to_z3(output_dir: str):
    """Load FOLIO dataset from HuggingFace and convert to Z3 code."""
    from datasets import load_dataset

    print("Downloading FOLIO dataset from HuggingFace...")
    ds = load_dataset('yale-nlp/FOLIO')

    output_records = []
    skipped = 0

    for split_name in ['train', 'validation']:
        for rec in ds[split_name]:
            premises_fol = rec.get('premises-FOL', '')
            conclusion_fol = rec.get('conclusion-FOL', '')
            premises_nl = rec.get('premises', '')
            conclusion_nl = rec.get('conclusion', '')
            label = rec.get('label', '')
            story_id = rec.get('story_id', 0)
            example_id = rec.get('example_id', 0)

            if not premises_fol or not conclusion_fol:
                skipped += 1
                continue

            z3_code = fol_to_z3_code(
                premises_fol, conclusion_fol,
                premises_nl, conclusion_nl,
                label
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

            output_records.append({
                "instruction": instruction,
                "input": "",
                "output": z3_code,
                "source": "folio_z3",
                "id": f"folio_{story_id}_{example_id}"
            })

    converted = len(output_records)
    print(f"[FOLIO->Z3] {converted} converted, {skipped} skipped")

    # Save
    os.makedirs(output_dir, exist_ok=True)

    import random
    random.seed(42)
    random.shuffle(output_records)

    split_idx = int(len(output_records) * 0.9)
    train_records = output_records[:split_idx]
    val_records = output_records[split_idx:]

    train_path = os.path.join(output_dir, 'train.jsonl')
    val_path = os.path.join(output_dir, 'val.jsonl')

    with open(train_path, 'w', encoding='utf-8') as f:
        for rec in train_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    with open(val_path, 'w', encoding='utf-8') as f:
        for rec in val_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f"\n{'='*60}")
    print(f"Z3 DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Total:  {converted} records")
    print(f"  Train:  {len(train_records)} records -> {train_path}")
    print(f"  Val:    {len(val_records)} records -> {val_path}")
    print(f"{'='*60}")

    return output_records


def main():
    parser = argparse.ArgumentParser(description='Convert FOLIO FOL to Z3 code')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Output directory for train/val JSONL')
    args = parser.parse_args()

    convert_folio_to_z3(args.output_dir)


if __name__ == '__main__':
    main()
