"""
Convert physics problems (electro_dataset) to executable SymPy verification code.

Consolidated script combining generation, verification, error analysis, and preview.

Usage:
  python scripts/convert_physics_to_sympy.py --generate [--start N] [--count M]
  python scripts/convert_physics_to_sympy.py --verify
  python scripts/convert_physics_to_sympy.py --analyze
  python scripts/convert_physics_to_sympy.py --preview [N]

Modes:
  --generate  : Read electro_dataset.jsonl, generate SymPy code, write to electro_sympy_dataset.jsonl
  --verify    : Run ALL generated SymPy code and report pass/fail rates
  --analyze   : Categorize failure patterns in generated SymPy code
  --preview N : Preview the first N generated records (default 3)
"""

import json
import re
import argparse
import os
from collections import Counter


# ─── Default Paths ─────────────────────────────────────────────────────────

DEFAULT_INPUT = r'd:\Exact 2026\data\collected\electro_dataset.jsonl'
DEFAULT_OUTPUT = r'd:\Exact 2026\data\collected\electro_sympy_dataset.jsonl'


# ═══════════════════════════════════════════════════════════════════════════
# Part 1: LaTeX-to-SymPy Conversion Engine
# ═══════════════════════════════════════════════════════════════════════════

def _is_unconvertible_latex(s: str) -> bool:
    """Check if a LaTeX string contains constructs we can't convert."""
    unconvertible = [
        r'\begin{', r'\end{', r'\propto', r'\approx', r'\sim',
        r'\neq', r'\leq', r'\geq', r'\ll', r'\gg',
        r'\\\\',  # line breaks in aligned environments
        r'&',     # alignment markers
    ]
    for pattern in unconvertible:
        if pattern in s:
            return True
    if re.search(r'(?<!\\)\|', s):
        return True
    if re.search(r"[a-zA-Z]'+", s):
        return True
    depth = 0
    for ch in s:
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
        elif ch == ',' and depth == 0:
            return True
    return False


def latex_to_sympy_expr(latex_str: str) -> str:
    """
    Convert a LaTeX math expression to a SymPy-compatible Python expression.

    Heuristic converter handling the most common patterns in electro_dataset.
    Returns None for unconvertible expressions.
    """
    s = latex_str.strip()

    if _is_unconvertible_latex(s):
        return None

    s = re.sub(r'^\$+|\$+$', '', s)

    for tag in [r'\displaystyle', r'\textstyle']:
        s = s.replace(tag, '')

    s = s.replace(r'\left', '').replace(r'\right', '')
    s = re.sub(r'\\[,;:!]', ' ', s)
    s = s.replace(r'\qquad', ' ').replace(r'\quad', ' ')

    text_match = re.match(r'^\\text\{(.+)\}$', s.strip())
    if text_match:
        return f'"{text_match.group(1)}"'

    s = re.sub(r'\s*\\?quad\s*\\text\{\s*(?:at|for|when|if)\s*\}.*$', '', s)
    s = re.sub(r'\\text\{[^}]*\}', '', s)
    s = re.sub(r'\s+(?:at|for|when|if)\s+[a-zA-Z]\s*=.*$', '', s, flags=re.IGNORECASE)

    s = re.sub(r'_\{?-\}?', '_minus', s)
    s = re.sub(r'_\{?\+\}?', '_plus', s)
    s = re.sub(r'_\{\s*\}', '', s)

    s = re.sub(r'\\mathbf\{([^}]+)\}\s*\([^)]*\)', r'\1', s)
    s = re.sub(r'\\mathbf\{([^}]+)\}', r'\1', s)

    s = re.sub(r'\\?\be_\{?\\?[a-zA-Z]+\}?', '', s)
    s = re.sub(r'\\hat\{[^}]+\}', '', s)

    s = s.replace('[', '(').replace(']', ')')
    s = s.replace(r'\{', '(').replace(r'\}', ')')

    for _ in range(5):
        s = re.sub(r'\\sqrt\{([^{}]+)\}', r'sqrt(\1)', s)

    for _ in range(10):
        s = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'((\1)/(\2))', s)

    s = re.sub(r'\be\^\{([^{}]+)\}', r'exp(\1)', s)
    s = re.sub(r'\be\^\(([^()]+)\)', r'exp(\1)', s)

    s = re.sub(r'\^\{([^{}]+)\}', r'**(\1)', s)
    s = re.sub(r'\^([0-9a-zA-Z])', r'**\1', s)

    for func in ['cos', 'sin', 'tan']:
        s = re.sub(
            rf'\b{func}\*\*\(?([0-9]+)\)?\s*\(([^)]+)\)',
            rf'{func}(\2)**(\1)', s
        )
        s = re.sub(
            rf'\b{func}\*\*\(?([0-9]+)\)?\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            rf'{func}(\2)**(\1)', s
        )

    if r'\cdots' in s or r'\ldots' in s or r'\dots' in s:
        return None
    s = s.replace(r'\cdot', '*')
    s = s.replace(r'\times', '*')
    s = s.replace(r'\pm', '+')
    s = s.replace(r'\mp', '-')
    s = s.replace(r'\ln', 'log')

    for func in ['cos', 'sin', 'tan', 'exp', 'arccos', 'arcsin', 'arctan']:
        s = s.replace(f'\\{func}', f'{func}')

    s = s.replace(r'\pi', 'pi')
    s = s.replace(r'\infty', 'oo')

    if r'\rightarrow' in s:
        return None

    # Common physics symbols
    replacements = {
        r'\epsilon_0': 'epsilon_0', r'\varepsilon_0': 'epsilon_0',
        r'\epsilon': 'epsilon', r'\varepsilon': 'epsilon',
        r'\mu_0': 'mu_0', r'\mu': 'mu',
        r'\sigma': 'sigma', r'\omega': 'omega', r'\theta': 'theta',
        r'\rho': 'rho', r'\lambda': 'lambda_', r'\phi': 'phi',
        r'\Phi': 'Phi', r'\Delta': 'Delta', r'\nabla': 'nabla',
    }
    for latex, py in replacements.items():
        s = s.replace(latex, py)

    s = re.sub(r'_\{([^{}]+)\}', r'_\1', s)
    s = s.replace('{', '(').replace('}', ')')
    s = re.sub(r'\\([a-zA-Z]+)', r'\1', s)
    s = re.sub(r'\s+', ' ', s).strip()

    for func in ['cos', 'sin', 'tan', 'exp', 'log', 'arccos', 'arcsin', 'arctan']:
        s = re.sub(
            rf'\b{func}\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            rf'{func}(\1)', s
        )

    if re.search(r'(?<!\*)>(?!=)', s) or re.search(r'(?<!\*)<(?!=)', s):
        return None

    s = re.sub(r'\*[a-zA-Z_]\w*=[a-zA-Z0-9_.]+\s*$', '', s)

    if '=' in s:
        return None

    s = re.sub(r'\bfrac\*?\(([^)]+)\)\*?\(([^)]+)\)', r'((\1)/(\2))', s)
    s = re.sub(r'\be\*\*\(([^)]+)\)', r'exp(\1)', s)
    s = re.sub(r'\be\^\(([^)]+)\)', r'exp(\1)', s)

    s_clean = _insert_implicit_multiplication(s)
    s_clean = s_clean.strip()
    if not s_clean or s_clean in ('', '*', '+', '-'):
        return None

    return s_clean


# ─── Implicit Multiplication ──────────────────────────────────────────────

def _insert_implicit_multiplication(s: str) -> str:
    """Insert * for implicit multiplication in a math expression string."""
    FUNCTIONS = {'sqrt', 'log', 'cos', 'sin', 'tan', 'exp', 'symbols', 'simplify',
                 'expand', 'factor', 'cancel', 'trigsimp', 'radsimp', 'abs'}
    result = []
    tokens = _tokenize_expr(s)

    for idx in range(len(tokens)):
        result.append(tokens[idx])
        if idx + 1 < len(tokens):
            if _needs_multiply(tokens[idx], tokens[idx + 1], FUNCTIONS):
                result.append('*')

    return ''.join(result)


def _tokenize_expr(s: str) -> list:
    """Tokenize a math expression into meaningful tokens."""
    tokens = []
    i = 0
    while i < len(s):
        if s[i].isspace():
            tokens.append(' ')
            while i < len(s) and s[i].isspace():
                i += 1
        elif s[i].isdigit() or (s[i] == '.' and i+1 < len(s) and s[i+1].isdigit()):
            j = i
            while j < len(s) and (s[j].isdigit() or s[j] == '.'):
                j += 1
            tokens.append(s[i:j])
            i = j
        elif s[i].isalpha() or s[i] == '_':
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] == '_'):
                j += 1
            tokens.append(s[i:j])
            i = j
        else:
            tokens.append(s[i])
            i += 1
    return [t for t in tokens if t.strip()]


def _needs_multiply(curr: str, nxt: str, functions: set) -> bool:
    """Determine if we need to insert * between two adjacent tokens."""
    curr_is_num = bool(re.match(r'^[\d.]+$', curr))
    curr_is_id = bool(re.match(r'^[a-zA-Z_]\w*$', curr))
    curr_is_close = curr in (')', ']')

    nxt_is_num = bool(re.match(r'^[\d.]+$', nxt))
    nxt_is_id = bool(re.match(r'^[a-zA-Z_]\w*$', nxt))
    nxt_is_open = nxt == '('

    if curr_is_num and nxt_is_id:
        return True
    if curr_is_id and nxt_is_num:
        return True
    if curr_is_id and nxt_is_id:
        return True
    if curr_is_close and nxt_is_open:
        return True
    if curr_is_close and nxt_is_id:
        return True
    if curr_is_close and nxt_is_num:
        return True
    if curr_is_num and nxt_is_open:
        return True
    if curr_is_id and nxt_is_open and curr not in functions:
        return True

    return False


# ═══════════════════════════════════════════════════════════════════════════
# Part 2: Code Generation
# ═══════════════════════════════════════════════════════════════════════════

def extract_equation_parts(answer_str: str):
    """Split a final_answer string into LHS and RHS at '='."""
    if r'\rightarrow' in answer_str or '->' in answer_str:
        return None, answer_str
    parts = answer_str.split('=', 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return None, answer_str.strip()


def generate_sympy_code(record: dict) -> str:
    """Generate a SymPy verification script for one dataset record."""
    problem_id = record['id']
    answers = record['final_answers']

    lines = []
    lines.append(f'"""Verification for problem {problem_id}"""')
    lines.append('from sympy import *')
    lines.append('')

    all_symbols = set()

    parsed_answers = []
    for ans in answers:
        lhs_latex, rhs_latex = extract_equation_parts(ans)
        lhs_py = latex_to_sympy_expr(lhs_latex) if lhs_latex else None
        rhs_py = latex_to_sympy_expr(rhs_latex)
        parsed_answers.append((ans, lhs_py, rhs_py))

        for expr_str in [lhs_py, rhs_py]:
            if expr_str and not expr_str.startswith('"'):
                tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', expr_str)
                known = {
                    'sqrt', 'log', 'cos', 'sin', 'tan', 'exp', 'pi', 'oo',
                    'Rational', 'Symbol', 'symbols', 'simplify', 'expand',
                    'factor', 'cancel', 'trigsimp', 'radsimp', 'E', 'I',
                    'None', 'True', 'False', 'print', 'abs',
                }
                for t in tokens:
                    if t not in known and not t.isdigit():
                        all_symbols.add(t)

    if all_symbols:
        sym_list = ', '.join(sorted(all_symbols))
        lines.append(f'# Define symbols')
        lines.append(f'{sym_list} = symbols("{sym_list}", positive=True)')
        lines.append('')

    lines.append('# Verify each final answer')
    lines.append('results = []')
    lines.append('')

    for i, (orig, lhs_py, rhs_py) in enumerate(parsed_answers):
        orig_safe = orig.replace('\\', '/').replace('\n', ' ').replace('\r', ' ').replace('"', "'")[:80]
        lines.append(f'# Answer {i+1}: {orig_safe}...')

        if rhs_py is None and lhs_py is None:
            lines.append(f'# SKIP: Complex LaTeX, cannot auto-convert')
            lines.append(f'results.append(("Answer {i+1}", "SKIP", "complex_latex"))')
        elif rhs_py and rhs_py.startswith('"'):
            lines.append(f'# Textual answer: {rhs_py}')
            lines.append(f'results.append(("Answer {i+1}", "TEXTUAL", {rhs_py}))')
        elif rhs_py is None:
            lines.append(f'# SKIP: RHS contains unconvertible LaTeX')
            lines.append(f'results.append(("Answer {i+1}", "SKIP", "unconvertible_rhs"))')
        elif lhs_py and rhs_py:
            lines.append(f'lhs_{i+1} = {lhs_py}')
            lines.append(f'rhs_{i+1} = {rhs_py}')
            lines.append(f'check_{i+1} = simplify(lhs_{i+1} - rhs_{i+1})')
            lines.append(f'results.append(("Answer {i+1}", "PASS" if check_{i+1} == 0 else "VERIFY", str(check_{i+1})))')
        else:
            if rhs_py:
                lines.append(f'expr_{i+1} = {rhs_py}')
                lines.append(f'results.append(("Answer {i+1}", "EXPR", str(expr_{i+1})))')
            else:
                lines.append(f'# SKIP: Empty expression after conversion')
                lines.append(f'results.append(("Answer {i+1}", "SKIP", "empty_expr"))')

        lines.append('')

    lines.append('# Print verification summary')
    lines.append('for name, status, detail in results:')
    lines.append('    print(f"  {name}: [{status}] {detail}")')

    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Part 3: Modes (generate / verify / analyze / preview)
# ═══════════════════════════════════════════════════════════════════════════

def mode_generate(args):
    """Generate SymPy verification code for each problem."""
    with open(args.input, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total = len(lines)
    end = min(args.start + args.count, total)
    print(f"Processing records {args.start} to {end-1} (total: {total})")

    output_records = []
    errors = []

    for idx in range(args.start, end):
        record = json.loads(lines[idx])
        problem_id = record['id']
        print(f"\n{'='*60}")
        print(f"[{idx}] {problem_id}")

        try:
            sympy_code = generate_sympy_code(record)
            record['sympy_verify_code'] = sympy_code
            output_records.append(record)
            print(f"  [OK] Generated {len(sympy_code)} chars of SymPy code")
        except Exception as e:
            errors.append((idx, problem_id, str(e)))
            print(f"  [ERR] Error: {e}")
            record['sympy_verify_code'] = f"# ERROR: {e}"
            output_records.append(record)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for rec in output_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f"\n{'='*60}")
    print(f"Done! Wrote {len(output_records)} records to {args.output}")
    if errors:
        print(f"Errors: {len(errors)}")
        for idx, pid, err in errors:
            print(f"  [{idx}] {pid}: {err}")


def mode_verify(args):
    """Run ALL generated SymPy code and report pass/fail rates."""
    with open(args.output, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f]

    total = len(records)
    print(f"Total records: {total}\n")

    passed = 0
    failed = 0
    fail_list = []

    for rec in records:
        code = rec['sympy_verify_code']
        pid = rec['id']
        try:
            exec(code, {'__builtins__': __builtins__})
            passed += 1
        except Exception as e:
            failed += 1
            fail_list.append((pid, type(e).__name__, str(e)[:80]))

    print()
    print("=" * 60)
    pct = passed / total * 100 if total else 0
    print(f"Results: {passed}/{total} passed ({pct:.1f}%)")
    print(f"         {failed}/{total} failed ({100-pct:.1f}%)")
    print("=" * 60)

    if fail_list:
        print(f"\nFailed records ({len(fail_list)}):")
        for pid, etype, msg in fail_list:
            print(f"  [{etype}] {pid}: {msg}")


def mode_analyze(args):
    """Categorize failure patterns in generated SymPy code."""
    with open(args.output, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f]

    error_patterns = Counter()
    error_details = []

    for rec in records:
        code = rec['sympy_verify_code']
        pid = rec['id']
        try:
            exec(code, {'__builtins__': __builtins__})
        except SyntaxError as e:
            code_lines = code.split('\n')
            bad_line = code_lines[e.lineno - 1].strip() if e.lineno and e.lineno <= len(code_lines) else "???"

            if "unterminated string literal" in str(e):
                error_patterns["unterminated_string"] += 1
            elif "'{' was never closed" in str(e):
                error_patterns["unclosed_brace"] += 1
            elif "cannot assign to expression" in str(e):
                error_patterns["assign_to_expr"] += 1
            elif "unexpected character after line continuation" in str(e):
                error_patterns["line_continuation"] += 1
            elif "unmatched" in str(e):
                error_patterns["unmatched_bracket"] += 1
            elif "forgot a comma" in str(e):
                error_patterns["forgot_comma"] += 1
            else:
                error_patterns["other_syntax"] += 1
            error_details.append((pid, str(e)[:60], bad_line[:100]))
        except TypeError as e:
            msg = str(e)
            if "FunctionClass" in msg:
                error_patterns["func_class_pow"] += 1
            elif "StrictGreaterThan" in msg or "StrictLessThan" in msg:
                error_patterns["comparison_op"] += 1
            else:
                error_patterns["other_type"] += 1
            error_details.append((pid, msg[:60], ""))
        except Exception as e:
            error_patterns[f"other_{type(e).__name__}"] += 1

    print("Error pattern breakdown:")
    for pattern, count in error_patterns.most_common():
        print(f"  {count:3d}  {pattern}")

    print(f"\nSample problematic lines (first 15):")
    for pid, err, line in error_details[:15]:
        print(f"  {pid}: {err}")
        if line:
            print(f"    LINE: {line}")


def mode_preview(args):
    """Preview the first N generated records."""
    n = args.preview_count

    with open(args.output, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            rec = json.loads(line)
            print(f"{'='*70}")
            print(f"Problem: {rec['id']}")
            print(f"{'='*70}")
            print(rec['sympy_verify_code'])
            print()


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Convert physics problems to SymPy verification code',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --generate --count 242          Generate all records
  %(prog)s --generate --start 10 --count 5 Generate 5 records starting from index 10
  %(prog)s --verify                        Verify all generated code runs
  %(prog)s --analyze                       Analyze error patterns
  %(prog)s --preview 5                     Preview first 5 records
        """
    )

    # Modes (mutually exclusive)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--generate', action='store_true', help='Generate SymPy code from electro_dataset')
    mode.add_argument('--verify', action='store_true', help='Verify all generated code executes')
    mode.add_argument('--analyze', action='store_true', help='Analyze error patterns in generated code')
    mode.add_argument('--preview', type=int, nargs='?', const=3, dest='preview_count',
                      help='Preview first N records (default 3)')

    # Generation options
    parser.add_argument('--start', type=int, default=0, help='Start index (0-based, for --generate)')
    parser.add_argument('--count', type=int, default=5, help='Number of records (for --generate)')
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT, help='Input JSONL path')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT, help='Output JSONL path')

    args = parser.parse_args()

    if args.generate:
        mode_generate(args)
    elif args.verify:
        mode_verify(args)
    elif args.analyze:
        mode_analyze(args)
    elif args.preview_count is not None:
        mode_preview(args)


if __name__ == '__main__':
    main()
