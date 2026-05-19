"""Verify SymPy code trong physics_kb.raw.jsonl, mark verified=true/false.

Mirror behavior cua physics_solver_node: subprocess timeout 10s, capture stdout/stderr.
Filter: chi giu record verified=True khi index sang Qdrant (lam o build_physics_index).

Usage:
    python -m scripts.distill.verify_kb
    python -m scripts.distill.verify_kb --in custom.raw.jsonl --out custom.verified.jsonl
    python -m scripts.distill.verify_kb --timeout 5
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.config import settings  # noqa: E402
from src.distillation.schema import KBRecord  # noqa: E402
from src.utils.logger import logger  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TIMEOUT_S = 10


def _run_code(code: str, timeout_s: int) -> tuple[bool, str, str]:
    """Run code trong subprocess Python rieng. Tra ve (ok, stdout, stderr)."""
    if not code.strip():
        return False, "", "empty code"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            encoding="utf-8",
            errors="replace",
        )
        ok = proc.returncode == 0
        return ok, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired:
        return False, "", f"TimeoutError: code chay qua {timeout_s}s"
    except Exception as e:  # noqa: BLE001
        return False, "", f"{type(e).__name__}: {e}"
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def _verify_one(rec: KBRecord, timeout_s: int) -> KBRecord:
    ok, stdout, stderr = _run_code(rec.sympy_code, timeout_s)
    rec.verified = bool(ok and "FINAL_ANSWER" in stdout)
    rec.exec_output = stdout[:2000]
    rec.exec_error = "" if ok else stderr[:2000]
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify SymPy code in distilled KB")
    parser.add_argument("--in", dest="in_path", type=str, default="",
                        help="Input raw KB jsonl (default: tu setting.yaml)")
    parser.add_argument("--out", dest="out_path", type=str, default="",
                        help="Output verified KB jsonl (default: tu setting.yaml)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S,
                        help=f"Per-record timeout (default {DEFAULT_TIMEOUT_S}s)")
    args = parser.parse_args()

    cfg = settings.distillation
    in_path = Path(args.in_path or PROJECT_ROOT / cfg.paths.raw_output)
    out_path = Path(args.out_path or PROJECT_ROOT / cfg.paths.verified_output)

    if not in_path.exists():
        raise FileNotFoundError(f"Raw KB not found: {in_path}")

    print("=" * 70)
    print(f"Verify SymPy code")
    print(f"Input  : {in_path}")
    print(f"Output : {out_path}")
    print(f"Timeout: {args.timeout}s per record")
    print("=" * 70)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    n_pass = 0
    n_fail = 0
    by_topic_pass: Counter[str] = Counter()
    by_topic_fail: Counter[str] = Counter()

    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                rec = KBRecord.from_jsonl(line)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"  [skip] malformed line: {e}")
                continue

            verified = _verify_one(rec, args.timeout)
            fout.write(verified.to_jsonl() + "\n")

            if verified.verified:
                n_pass += 1
                by_topic_pass[verified.topic] += 1
            else:
                n_fail += 1
                by_topic_fail[verified.topic] += 1

            if n_total % 50 == 0:
                print(f"  [{n_total}] pass={n_pass}  fail={n_fail}")

    print()
    print("=" * 70)
    pct = (n_pass / n_total * 100) if n_total else 0
    print(f"Total : {n_total}")
    print(f"Pass  : {n_pass} ({pct:.1f}%)")
    print(f"Fail  : {n_fail} ({100 - pct:.1f}%)")
    print()
    print("Pass by topic:")
    for t, c in by_topic_pass.most_common():
        print(f"  {c:4d}  {t}")
    if by_topic_fail:
        print()
        print("Fail by topic:")
        for t, c in by_topic_fail.most_common():
            print(f"  {c:4d}  {t}")
    print("=" * 70)


if __name__ == "__main__":
    main()
