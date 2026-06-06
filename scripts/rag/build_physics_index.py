"""Build 2 Qdrant collection cho physics RAG tu distilled KB.

Source: data/distilled/physics_kb.verified.jsonl (chi record verified=True).

2 collection xay dung:
1. physics_examples - per-record:
   text = "Problem ... \n Topic ... \n Formulas ... \n Code ..."
   Dung de tim bai toan giong runtime question (semantic match).

2. physics_formulas - per-topic:
   text = "Topic ... \n Formulas tong hop tu tat ca record cung topic"
   Dung de fallback khi khong co bai giong: cap formula sheet theo topic.

Idempotent: rebuild bang cach xoa storage/qdrant_storage/ truoc khi chay.

Usage:
    python -m scripts.rag.build_physics_index
    python -m scripts.rag.build_physics_index --rebuild
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.config import settings  # noqa: E402
from src.utils.logger import logger  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Collection names - physics_rag_node.py phai khop.
COLLECTION_EXAMPLES = "physics_examples"
COLLECTION_FORMULAS = "physics_formulas"


@dataclass
class KBRecord:
    """Minimal, provider-independent schema consumed by the RAG indexer."""

    id: str = ""
    source: str = ""
    problem: str = ""
    topic: str = "other"
    formulas: list[str] = field(default_factory=list)
    symbols: dict[str, str] = field(default_factory=dict)
    sympy_code: str = ""
    answer: str = ""
    verified: bool = False

    @classmethod
    def from_jsonl(cls, line: str) -> "KBRecord":
        data = json.loads(line)
        formulas = data.get("formulas") or []
        if isinstance(formulas, str):
            formulas = [formulas]
        symbols = data.get("symbols") or {}
        return cls(
            id=str(data.get("id") or ""),
            source=str(data.get("source") or ""),
            problem=str(data.get("problem") or data.get("question") or ""),
            topic=str(data.get("topic") or "other"),
            formulas=[str(item) for item in formulas],
            symbols={str(key): str(value) for key, value in dict(symbols).items()},
            sympy_code=str(data.get("sympy_code") or ""),
            answer=str(data.get("answer") or ""),
            verified=data.get("verified") is True,
        )


def _format_example_text(rec: KBRecord) -> str:
    """Format 1 record cho per-record collection (problem + formulas + code).

    Embed se nhin vao toan bo block nay -> matching theo problem + formulas + code style.
    """
    formulas_block = "\n".join(f"  - {f}" for f in rec.formulas) if rec.formulas else "  (none)"
    symbols_block = "\n".join(
        f"  - {sym}: {desc}" for sym, desc in (rec.symbols or {}).items()
    ) if rec.symbols else "  (none)"

    return (
        f"Topic: {rec.topic}\n"
        f"Problem:\n{rec.problem}\n\n"
        f"Formulas:\n{formulas_block}\n\n"
        f"Symbols:\n{symbols_block}\n\n"
        f"SymPy code:\n```python\n{rec.sympy_code}\n```\n\n"
        f"Answer: {rec.answer}"
    )


def _format_topic_text(topic: str, formulas_set: list[str], symbols_map: dict[str, str]) -> str:
    """Format formula sheet cho 1 topic (gop tu nhieu record).

    Dung khi runtime question khong match bai cu the nao -> fallback bang formula sheet
    cua topic gan nhat semantic.
    """
    formulas_block = "\n".join(f"  - {f}" for f in formulas_set) if formulas_set else "  (none)"
    symbols_block = "\n".join(
        f"  - {sym}: {desc}" for sym, desc in sorted(symbols_map.items())
    ) if symbols_map else "  (none)"

    return (
        f"Topic: {topic}\n\n"
        f"Canonical formulas (deduplicated):\n{formulas_block}\n\n"
        f"Common symbols:\n{symbols_block}"
    )


def _load_verified(paths: list[Path]) -> list[KBRecord]:
    """Load từ một hay nhiều file JSONL, chỉ giữ record verified=True.

    Q19: Also filters out records with IDs starting with 'QA' (401 annotation
    errors in the Physics dataset that BTC confirmed should be excluded).
    """
    out: list[KBRecord] = []
    n_qa_filtered = 0
    for path in paths:
        if not path.exists():
            logger.warning(f"  [skip] not found: {path}")
            continue
        n_kept = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = KBRecord.from_jsonl(line)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"  [skip] malformed line in {path.name}: {e}")
                    continue
                # Q19: Filter out QA-prefixed annotation errors
                if rec.id and str(rec.id).upper().startswith("QA"):
                    n_qa_filtered += 1
                    continue
                if rec.verified is True:
                    out.append(rec)
                    n_kept += 1
        logger.info(f"  loaded {n_kept} verified records from {path.name}")
    if n_qa_filtered > 0:
        logger.info(f"  [Q19] Filtered out {n_qa_filtered} QA-prefixed records.")
    return out


def _build_examples_collection(records: list[KBRecord], vdb) -> None:
    """Build per-record collection (mot doc / record)."""
    from llama_index.core import Document

    docs = [
        Document(
            text=_format_example_text(rec),
            metadata={
                "id": rec.id,
                "topic": rec.topic,
                "source": rec.source,
                "answer": rec.answer,
            },
        )
        for rec in records
    ]
    if not docs:
        logger.warning(f"No verified records -> skipping {COLLECTION_EXAMPLES}")
        return

    vdb.add_documents(documents=docs, collection_name=COLLECTION_EXAMPLES)
    logger.info(f"OK: built '{COLLECTION_EXAMPLES}' voi {len(docs)} examples.")


def _build_formulas_collection(records: list[KBRecord], vdb) -> None:
    """Build per-topic collection (mot doc / topic, gop formula).

    Dedup formula bang lower-case + strip whitespace.
    """
    from llama_index.core import Document

    by_topic: dict[str, list[KBRecord]] = defaultdict(list)
    for rec in records:
        by_topic[rec.topic or "other"].append(rec)

    docs: list[Document] = []
    for topic, recs in by_topic.items():
        # Dedup formulas
        seen: set[str] = set()
        unique_formulas: list[str] = []
        for r in recs:
            for f in r.formulas:
                key = f.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    unique_formulas.append(f.strip())

        # Merge symbols (later wins)
        merged_symbols: dict[str, str] = {}
        for r in recs:
            for k, v in (r.symbols or {}).items():
                if k not in merged_symbols:
                    merged_symbols[k] = v

        text = _format_topic_text(topic, unique_formulas, merged_symbols)
        docs.append(
            Document(
                text=text,
                metadata={
                    "topic": topic,
                    "n_records": len(recs),
                    "n_formulas": len(unique_formulas),
                },
            )
        )

    if not docs:
        logger.warning(f"No topics -> skipping {COLLECTION_FORMULAS}")
        return

    vdb.add_documents(documents=docs, collection_name=COLLECTION_FORMULAS)
    logger.info(
        f"OK: built '{COLLECTION_FORMULAS}' voi {len(docs)} topic blocks "
        f"(total {sum(d.metadata['n_formulas'] for d in docs)} formulas)."
    )


def _resolve_inputs(args_inputs: list[str]) -> list[Path]:
    """Quyết định input files.

    Ưức tiên: --input flag (có thể truyền nhiều lần).
    Fallback: tự dò trong data/distilled/ theo thứ tự ưu tiên.
    """
    if args_inputs:
        return [Path(p) for p in args_inputs]

    candidates = [
        PROJECT_ROOT / settings.distillation.paths.verified_output,
        PROJECT_ROOT / "data" / "distilled" / "physics_kb.formulas.jsonl",
        PROJECT_ROOT / "data" / "distilled" / "physics_kb.from_pf.jsonl",
    ]
    found = [p for p in candidates if p.exists()]
    if not found:
        raise FileNotFoundError(
            "No verified physics KB JSONL was found. Pass one or more "
            "--input paths containing disclosed, verified records."
        )
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="Build physics RAG index tu distilled KB")
    parser.add_argument("--input", action="append", default=[],
                        help="Input KB jsonl (truyen nhieu lan de merge). "
                             "Default: auto-detect verified JSONL in data/distilled/")
    parser.add_argument("--rebuild", action="store_true",
                        help="Xoa storage/qdrant_storage truoc khi build (idempotent)")
    args = parser.parse_args()

    input_paths = _resolve_inputs(args.input)

    if args.rebuild:
        qdrant_dir = PROJECT_ROOT / "storage" / "qdrant_storage"
        if qdrant_dir.exists():
            shutil.rmtree(qdrant_dir, ignore_errors=True)
            logger.info(f"Removed {qdrant_dir} (rebuild mode)")
        for col in (COLLECTION_EXAMPLES, COLLECTION_FORMULAS):
            persist = PROJECT_ROOT / "storage" / col
            if persist.exists():
                shutil.rmtree(persist, ignore_errors=True)

    print("=" * 70)
    print("Build physics RAG index")
    for p in input_paths:
        print(f"Input: {p}")
    print("=" * 70)

    records = _load_verified(input_paths)
    print(f"Loaded {len(records)} verified records total")
    if not records:
        print("FAIL: 0 verified records. Khong build.")
        sys.exit(1)

    # Single VectorDBManager instance to avoid Qdrant file lock conflicts
    from src.agent.llm.embedding import EmbeddingFactory
    from src.retrieval.vector_db import VectorDBManager

    embed = EmbeddingFactory().get_embedding()
    vdb = VectorDBManager(embedding_model=embed)

    print("\n[1/2] Building per-record examples collection...")
    _build_examples_collection(records, vdb)

    print("\n[2/2] Building per-topic formulas collection...")
    _build_formulas_collection(records, vdb)

    print("\nDone.")


if __name__ == "__main__":
    main()
