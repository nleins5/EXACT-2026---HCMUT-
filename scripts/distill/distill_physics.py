"""Distill physics knowledge from teacher LLM -> data/distilled/physics_kb.raw.jsonl.

Resumable: reads existing output, skips IDs already done.
Concurrent: uses asyncio.Semaphore to limit parallel requests.

Usage:
    python -m scripts.distill.distill_physics --source btc
    python -m scripts.distill.distill_physics --source electro --limit 10
    python -m scripts.distill.distill_physics --source all --concurrency 6

CLI:
    --source btc | electro | all  (default: all)
    --limit N                     (default: full)
    --concurrency K               (default: from setting.yaml)
    --output PATH                 (default: from setting.yaml)
    --dry-run                     (list IDs only, no LLM calls)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Allow importing src.* when running this module.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.data_prep._common import load_btc_physics, load_electro_sympy  # noqa: E402
from src.core.config import settings  # noqa: E402
from scripts.distill.schema import KBRecord  # noqa: E402
from scripts.distill.teacher_client import TeacherClientError, build_teacher_client  # noqa: E402
from src.utils.logger import logger  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ═══════════════════════════════════════════════════════════════════════════
# Source loaders -> normalize to (id, source, problem, hint, answer, unit)
# ═══════════════════════════════════════════════════════════════════════════


def _load_btc_problems() -> list[tuple[str, str, str, str, str, str]]:
    """BTC physics CSV -> list[(id, source, problem, hint, answer, unit)]."""
    qa_list = load_btc_physics()
    out: list[tuple[str, str, str, str, str, str]] = []
    for q in qa_list:
        rid = f"btc_{q.id}"
        out.append((rid, "btc_physics", q.question, q.cot or "", q.answer or "", q.unit or ""))
    return out


def _load_electro_problems() -> list[tuple[str, str, str, str, str, str]]:
    """electro_sympy_dataset.jsonl -> list[(id, source, problem, hint, answer, unit)]."""
    samples = load_electro_sympy()
    out: list[tuple[str, str, str, str, str, str]] = []
    for s in samples:
        rid = f"electro_{s.id}"
        # Hint = first paragraph of solution (truncate to avoid overloading prompt).
        hint = (s.solution or "")[:1000]
        # ElectroSample may not have unit field; default empty.
        ans = getattr(s, "answer", "") or ""
        unit = getattr(s, "unit", "") or ""
        out.append((rid, "electro", s.question, hint, ans, unit))
    return out


def _load_problems(source: str) -> list[tuple[str, str, str, str, str, str]]:
    if source == "btc":
        return _load_btc_problems()
    if source == "electro":
        return _load_electro_problems()
    if source == "all":
        return _load_btc_problems() + _load_electro_problems()
    raise ValueError(f"Unknown source: {source}")


# ═══════════════════════════════════════════════════════════════════════════
# Resumable I/O
# ═══════════════════════════════════════════════════════════════════════════


def _existing_ids(output_path: Path) -> set[str]:
    """Read existing output, return set of IDs already distilled."""
    if not output_path.exists():
        return set()
    seen: set[str] = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def _append_jsonl(path: Path, record: KBRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(record.to_jsonl() + "\n")


def _append_cost_log(path: Path, record_id: str, in_tok: int, out_tok: int, latency_s: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "id": record_id,
        "in_tokens": in_tok,
        "out_tokens": out_tok,
        "latency_s": round(latency_s, 2),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# Worker
# ═══════════════════════════════════════════════════════════════════════════


async def _distill_one(
    sem: asyncio.Semaphore,
    client,
    record_id: str,
    source: str,
    problem: str,
    hint: str,
    answer: str,
    unit: str,
    output_path: Path,
    cost_log_path: Path,
) -> tuple[str, bool, str]:
    """Distill 1 problem, write immediately to file (incremental). Returns (id, ok, error)."""
    async with sem:
        t0 = time.monotonic()
        try:
            kb = await client.distill_one(
                record_id=record_id,
                source=source,
                problem=problem,
                hint=hint,
                answer=answer,
                unit=unit,
            )
            latency = time.monotonic() - t0
            _append_jsonl(output_path, kb)
            _append_cost_log(cost_log_path, record_id, kb.input_tokens, kb.output_tokens, latency)
            return (record_id, True, "")
        except TeacherClientError as e:
            return (record_id, False, str(e))
        except Exception as e:  # noqa: BLE001
            return (record_id, False, f"{type(e).__name__}: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Main async
# ═══════════════════════════════════════════════════════════════════════════


async def _amain(args: argparse.Namespace) -> int:
    cfg = settings.distillation
    output_path = Path(args.output or PROJECT_ROOT / cfg.paths.raw_output)
    cost_log_path = PROJECT_ROOT / cfg.paths.cost_log
    concurrency = args.concurrency or cfg.pipeline.concurrency

    problems = _load_problems(args.source)
    seen = _existing_ids(output_path)
    pending = [p for p in problems if p[0] not in seen]
    if args.limit:
        pending = pending[: args.limit]

    print("=" * 70)
    print(f"Distillation source : {args.source}")
    print(f"Mode                : {cfg.pipeline.mode}")
    print(f"Total problems      : {len(problems)}")
    print(f"Already done        : {len(seen)}")
    print(f"To process this run : {len(pending)}")
    print(f"Output (append)     : {output_path}")
    print(f"Concurrency         : {concurrency}")
    print(f"Teacher             : {cfg.teacher.model_name}")
    print("=" * 70)

    if not pending:
        print("Nothing to do.")
        return 0

    if args.dry_run:
        print("\n[dry-run] First 10 IDs:")
        for p in pending[:10]:
            print(f"  - {p[0]}")
        return 0

    client = build_teacher_client()
    sem = asyncio.Semaphore(concurrency)

    tasks = [
        _distill_one(sem, client, rid, src, prob, hint, ans, unit, output_path, cost_log_path)
        for (rid, src, prob, hint, ans, unit) in pending
    ]

    n_ok = 0
    n_fail = 0
    failures: list[tuple[str, str]] = []

    for fut in asyncio.as_completed(tasks):
        rid, ok, err = await fut
        if ok:
            n_ok += 1
            if n_ok % 25 == 0:
                print(f"  [{n_ok + n_fail}/{len(pending)}] ok so far: {n_ok}")
        else:
            n_fail += 1
            failures.append((rid, err))
            logger.warning(f"  [fail] {rid}: {err}")

    print()
    print("=" * 70)
    print(f"Done. ok={n_ok}  fail={n_fail}  total={len(pending)}")
    if failures:
        print(f"First 10 failures:")
        for rid, err in failures[:10]:
            print(f"  - {rid}: {err}")
    print(f"Raw KB        : {output_path}")
    print(f"Cost log      : {cost_log_path}")
    print("=" * 70)
    return 0 if n_fail == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill physics KB from teacher LLM")
    parser.add_argument("--source", choices=["btc", "electro", "all"], default="all")
    parser.add_argument("--limit", type=int, default=0, help="Max records (0 = all)")
    parser.add_argument("--concurrency", type=int, default=0, help="0 = use config")
    parser.add_argument("--output", type=str, default="", help="Override output path")
    parser.add_argument("--dry-run", action="store_true", help="List IDs only, no LLM calls")
    args = parser.parse_args()

    rc = asyncio.run(_amain(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
