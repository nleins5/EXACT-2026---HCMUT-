"""Shared utilities for data prep scripts.

Loaders, converters, ChatML formatter, exec-verifier, JSONL writer, STATS.md generator.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import os
import random
import re
import sys
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

# ═══════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "data"

BTC_DIR = DATA_DIR / "EXACT2026_dataset_2026-05-15"
BTC_LOGIC_JSON = (
    BTC_DIR / "Logic_Based_Educational_Queries_Text_Only" / "Logic_Based_Educational_Queries.json"
)
BTC_PHYSICS_CSV = (
    BTC_DIR / "Physics_Problems_Text_Only" / "Physics_Problems_Text_Only.csv"
)
ELECTRO_JSONL = DATA_DIR / "collected" / "electro_dataset.jsonl"
ELECTRO_SYMPY_JSONL = DATA_DIR / "collected" / "electro_sympy_dataset.jsonl"

OUTPUT_DIR = DATA_DIR / "finetune"

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

SEED = 3407
VAL_RATIO = 0.1


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LogicQA:
    """One question-answer pair from BTC Logic (411 records → 808 pairs)."""
    record_idx: int
    q_idx: int
    premises_nl: list[str]
    premises_fol: list[str]
    question: str
    answer: str
    explanation: str

    @property
    def uid(self) -> str:
        return f"btc_logic_{self.record_idx:03d}_q{self.q_idx}"


@dataclass
class PhysicsQA:
    """One physics problem from BTC Physics CSV (1352 rows after Q19 filter)."""
    id: str
    question: str
    cot: str
    answer: str
    unit: str

    @property
    def final_answer(self) -> str:
        return f"{self.answer} {self.unit}".strip() if self.unit else self.answer


@dataclass
class ElectroSample:
    """One electromagnetism textbook problem with executable SymPy code."""
    id: str
    question: str
    solution: str
    final_answers: list[str]
    sympy_code: str


@dataclass
class FolioSample:
    """One FOLIO record with NL+FOL premises, NL+FOL conclusion, and label."""
    uid: str
    premises_nl: list[str]
    premises_fol: list[str]
    conclusion_nl: str
    conclusion_fol: str
    label: str  # "True" | "False" | "Unknown"


# ═══════════════════════════════════════════════════════════════════════════
# Engine loaders (reuse legacy converters)
# ═══════════════════════════════════════════════════════════════════════════

def _import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_z3_engine():
    """Lazy-load FOL→Z3 conversion engine from convert_logic_to_z3.py."""
    return _import_module(
        "convert_logic_to_z3",
        SCRIPTS_DIR / "convert_logic_to_z3.py",
    )


def get_sympy_engine():
    """Lazy-load LaTeX→SymPy conversion engine from convert_physics_to_sympy.py."""
    return _import_module(
        "convert_physics_to_sympy",
        SCRIPTS_DIR / "convert_physics_to_sympy.py",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Loaders
# ═══════════════════════════════════════════════════════════════════════════

def load_btc_logic() -> list[LogicQA]:
    """Load BTC Logic JSON, expand to one record per Q/A pair.

    Per QA.pdf Q19 (logic), no row filter required. We use idx[] to subset
    premises per question when present, else use all premises.
    """
    raw = json.loads(BTC_LOGIC_JSON.read_text(encoding="utf-8"))
    out: list[LogicQA] = []
    for rec_i, rec in enumerate(raw):
        prem_nl_all: list[str] = rec.get("premises-NL", [])
        prem_fol_all: list[str] = rec.get("premises-FOL", [])
        questions: list[str] = rec.get("questions", [])
        answers: list[str] = rec.get("answers", [])
        explanations: list[str] = rec.get("explanation", [])
        idx_map: list[list[int]] = rec.get("idx", [])

        for q_i, q in enumerate(questions):
            ans = answers[q_i] if q_i < len(answers) else None
            if not ans:
                continue
            expl = explanations[q_i] if q_i < len(explanations) else ""
            sel_indices = idx_map[q_i] if q_i < len(idx_map) and idx_map[q_i] else None

            if sel_indices:
                sel_nl: list[str] = []
                sel_fol: list[str] = []
                for one in sel_indices:
                    j = one - 1  # 1-based -> 0-based
                    if 0 <= j < len(prem_nl_all):
                        sel_nl.append(prem_nl_all[j])
                    if 0 <= j < len(prem_fol_all):
                        sel_fol.append(prem_fol_all[j])
            else:
                sel_nl = list(prem_nl_all)
                sel_fol = list(prem_fol_all)

            out.append(
                LogicQA(
                    record_idx=rec_i,
                    q_idx=q_i,
                    premises_nl=sel_nl,
                    premises_fol=sel_fol,
                    question=q,
                    answer=ans.strip(),
                    explanation=expl.strip(),
                )
            )
    return out


def load_btc_physics() -> list[PhysicsQA]:
    """Load BTC Physics CSV with Q19 filter (drop id_prefix='QA' if any).

    The 2026-05-15 release already removed those rows; we keep the filter
    defensively for future datasets.
    """
    out: list[PhysicsQA] = []
    with open(BTC_PHYSICS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("id", "").strip()
            if rid.startswith("QA"):  # Q19 filter
                continue
            q = row.get("question", "").strip()
            ans = row.get("answer", "").strip()
            if not q or not ans:
                continue
            out.append(
                PhysicsQA(
                    id=rid,
                    question=q,
                    cot=row.get("cot", "").strip(),
                    answer=ans,
                    unit=row.get("unit", "").strip(),
                )
            )
    return out


_BASE64_RE = re.compile(r"data:image/[a-zA-Z]+;base64,[A-Za-z0-9+/=\s]+", re.MULTILINE)
_MD_IMAGE_BASE64_RE = re.compile(r"!\[[^\]]*\]\(data:image/[^)]+\)")


def strip_base64_images(text: str | None) -> str:
    """Remove embedded base64 PNG/JPG payloads from text."""
    if not text:
        return ""
    text = _MD_IMAGE_BASE64_RE.sub("[image-removed]", text)
    text = _BASE64_RE.sub("[base64-removed]", text)
    return text


def load_electro_sympy() -> list[ElectroSample]:
    """Load electro_sympy_dataset.jsonl (already has sympy_verify_code).

    If the *_sympy_dataset.jsonl is missing, generate it on-the-fly from
    electro_dataset.jsonl using the SymPy engine.
    """
    if not ELECTRO_SYMPY_JSONL.exists():
        _generate_electro_sympy()

    out: list[ElectroSample] = []
    with open(ELECTRO_SYMPY_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            qs = r.get("questions", "")
            if isinstance(qs, list):
                qs = "\n".join(qs)
            sols = r.get("solutions", "")
            if isinstance(sols, list):
                sols = "\n".join(sols)
            code = r.get("sympy_verify_code", "")
            if not code or code.startswith("# ERROR"):
                continue
            out.append(
                ElectroSample(
                    id=r.get("id", "electro/?"),
                    question=strip_base64_images(qs),
                    solution=strip_base64_images(sols),
                    final_answers=r.get("final_answers", []) or [],
                    sympy_code=code,
                )
            )
    return out


def _generate_electro_sympy():
    """Generate electro_sympy_dataset.jsonl if missing (one-shot)."""
    print(f"  [info] {ELECTRO_SYMPY_JSONL.name} missing, generating...")
    sympy_eng = get_sympy_engine()
    out_lines: list[str] = []
    with open(ELECTRO_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            try:
                code = sympy_eng.generate_sympy_code(r)
            except Exception as e:
                code = f"# ERROR: {e}"
            r["sympy_verify_code"] = code
            out_lines.append(json.dumps(r, ensure_ascii=False))
    ELECTRO_SYMPY_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(ELECTRO_SYMPY_JSONL, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")


def load_folio() -> list[FolioSample]:
    """Load yale-nlp/FOLIO via the `datasets` library.

    Returns an empty list with a clear log if the dataset cannot be downloaded
    (offline / hub blocked) so the pipeline can still produce something.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        print(f"  [warn] datasets library unavailable ({e}); skipping FOLIO")
        return []

    try:
        ds = load_dataset("yale-nlp/FOLIO")
    except Exception as e:
        print(f"  [warn] cannot load FOLIO from hub ({e}); skipping")
        return []

    out: list[FolioSample] = []
    for split_name in ("train", "validation"):
        if split_name not in ds:
            continue
        for rec in ds[split_name]:
            prem_fol = (rec.get("premises-FOL") or "").strip()
            concl_fol = (rec.get("conclusion-FOL") or "").strip()
            prem_nl = rec.get("premises") or ""
            concl_nl = rec.get("conclusion") or ""
            label = rec.get("label") or ""
            if not prem_fol or not concl_fol:
                continue

            story_id = rec.get("story_id", 0)
            example_id = rec.get("example_id", 0)
            uid = f"folio_{split_name}_{story_id}_{example_id}"

            # premises arrive as a single string with newlines; split for display
            prem_nl_list = [p.strip() for p in str(prem_nl).split("\n") if p.strip()]
            prem_fol_list = [p.strip() for p in str(prem_fol).split("\n") if p.strip()]

            out.append(
                FolioSample(
                    uid=uid,
                    premises_nl=prem_nl_list,
                    premises_fol=prem_fol_list,
                    conclusion_nl=str(concl_nl).strip(),
                    conclusion_fol=concl_fol,
                    label=str(label).strip(),
                )
            )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Code conversion (Logic-FOL → Z3) using legacy engine
# ═══════════════════════════════════════════════════════════════════════════

def folio_to_z3(s: FolioSample) -> str:
    """Convert one FOLIO sample into executable Z3 Python.

    Reuses the legacy `fol_to_z3_code` engine which already handles full
    premises-FOL + conclusion-FOL inputs.
    """
    eng = get_z3_engine()
    return eng.fol_to_z3_code(
        "\n".join(s.premises_fol),
        s.conclusion_fol,
        "\n".join(s.premises_nl),
        s.conclusion_nl,
        s.label,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Exec verifier
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ExecResult:
    ok: bool
    stdout: str
    error: str  # empty if ok


def verify_python(code: str) -> ExecResult:
    """Run code in an isolated namespace, capture stdout, return result.

    No timeout (Windows lacks SIGALRM in the stdlib). Trust that the
    generated Z3 / SymPy snippets terminate quickly. Anything that raises
    is captured as an error string.
    """
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    ns: dict = {"__builtins__": __builtins__}
    try:
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            exec(code, ns)
    except Exception as e:
        return ExecResult(ok=False, stdout=out_buf.getvalue(), error=f"{type(e).__name__}: {e}")
    return ExecResult(ok=True, stdout=out_buf.getvalue(), error="")


# ═══════════════════════════════════════════════════════════════════════════
# ChatML formatter
# ═══════════════════════════════════════════════════════════════════════════

def chatml(system: str, user: str, assistant: str, *, meta: dict | None = None) -> dict:
    """Build a ChatML record compatible with Unsloth/Qwen training.

    The schema is `{messages: [...], meta: {...}}` which is the format
    Unsloth's `tokenizer.apply_chat_template` ingests directly.
    """
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "meta": meta or {},
    }


# ═══════════════════════════════════════════════════════════════════════════
# I/O & splits
# ═══════════════════════════════════════════════════════════════════════════

def write_jsonl(records: Iterable[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"  -> {n:5d} samples  {path}")
    return n


def train_val_split(
    records: list[dict],
    val_ratio: float = VAL_RATIO,
    seed: int = SEED,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


def write_stats_md(records: list[dict], path: Path, *, title: str):
    """Emit a STATS.md table summarizing the dataset by source/type/branch."""
    from collections import Counter

    by_source: Counter[str] = Counter()
    by_type: Counter[str] = Counter()
    by_branch: Counter[str] = Counter()
    msg_lens: list[int] = []

    for r in records:
        meta = r.get("meta", {}) or {}
        by_source[meta.get("source", "?")] += 1
        by_type[meta.get("type", "?")] += 1
        by_branch[meta.get("branch", "?")] += 1
        msgs = r.get("messages", [])
        msg_lens.append(sum(len(m.get("content", "")) for m in msgs))

    def _table(c: Counter) -> str:
        if not c:
            return "(empty)"
        rows = ["| Key | Count | % |", "|---|---:|---:|"]
        total = sum(c.values()) or 1
        for k, v in c.most_common():
            rows.append(f"| `{k}` | {v} | {v/total*100:.1f} |")
        return "\n".join(rows)

    avg_len = sum(msg_lens) / len(msg_lens) if msg_lens else 0
    p95_len = sorted(msg_lens)[int(len(msg_lens) * 0.95) - 1] if msg_lens else 0

    md = f"""# {title}

Total samples: **{len(records)}**

Average chars per record: **{avg_len:.0f}**
P95 chars per record: **{p95_len}**

## By source
{_table(by_source)}

## By type
{_table(by_type)}

## By branch (success/error)
{_table(by_branch)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")
    print(f"  -> stats     {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def format_premises_block(premises_nl: list[str], premises_fol: list[str] | None = None) -> str:
    """Render premises as a numbered NL block, optionally appending FOL."""
    parts = []
    for i, p in enumerate(premises_nl, 1):
        parts.append(f"{i}. {p}")
    if premises_fol:
        parts.append("")
        parts.append("Formal logic (FOL):")
        for i, p in enumerate(premises_fol, 1):
            parts.append(f"  P{i}: {p}")
    return "\n".join(parts)


def truncate(text: str, n: int = 4000) -> str:
    """Hard-truncate long fields to keep ChatML records bounded."""
    if not text:
        return ""
    if len(text) <= n:
        return text
    return text[: n - 20] + "…[truncated]"
