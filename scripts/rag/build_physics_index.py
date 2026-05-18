"""Build Qdrant index cho physics few-shot examples.

Doc data/finetune/coder.jsonl, loc rows physics (system prompt mention SymPy
hoac source field bat dau bang `physics_` / `electro_`), embed bang BAAI/bge-m3,
luu vao collection `physics_examples` o storage/qdrant_storage/.

Chay 1 lan duy nhat khi muon enable RAG cho physics_formalizer:
    python -m scripts.rag.build_physics_index

Idempotent: neu collection da ton tai, doc lai metadata; muon rebuild thi
xoa storage/qdrant_storage/ truoc khi chay lai.
"""
from __future__ import annotations

import json
from pathlib import Path

from llama_index.core import Document

from src.agent.llm.embedding import EmbeddingFactory
from src.agent.nodes.physics_rag import PHYSICS_COLLECTION
from src.retrieval.vector_db import VectorDBManager
from src.utils.logger import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODER_JSONL = PROJECT_ROOT / "data" / "finetune" / "coder.jsonl"


def _is_physics_row(row: dict) -> bool:
    """Phan loai row co phai physics khong dua vao system prompt + user prompt.

    Coder dataset gom 2 nhom:
    - Logic: system prompt nhac den Z3.
    - Physics: system prompt nhac den SymPy.
    """
    msgs = row.get("messages", [])
    if not msgs:
        return False
    sys_msg = msgs[0].get("content", "") if msgs[0].get("role") == "system" else ""
    return "sympy" in sys_msg.lower() or "SymPy" in sys_msg


def _row_to_text(row: dict) -> str:
    """Trich (problem, code) tu 1 row coder.jsonl thanh chuoi few-shot."""
    msgs = row.get("messages", [])
    user_msg = next((m["content"] for m in msgs if m.get("role") == "user"), "")
    asst_msg = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")

    # User msg co the bao gom prefix `[PHYSICS PROBLEM]` -> giu nguyen, dung cho semantic match.
    return f"Problem:\n{user_msg.strip()}\n\nCode:\n{asst_msg.strip()}"


def main() -> None:
    if not CODER_JSONL.exists():
        raise FileNotFoundError(f"coder.jsonl khong ton tai: {CODER_JSONL}")

    logger.info(f"Reading {CODER_JSONL}...")
    docs: list[Document] = []
    n_total = 0
    with CODER_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            row = json.loads(line)
            if not _is_physics_row(row):
                continue
            docs.append(Document(text=_row_to_text(row)))

    logger.info(f"Loaded {len(docs)}/{n_total} physics examples tu coder.jsonl")
    if not docs:
        logger.warning("Khong co physics example nao -> bo qua build index.")
        return

    embed = EmbeddingFactory().get_embedding()
    vdb = VectorDBManager(embedding_model=embed)
    vdb.add_documents(documents=docs, collection_name=PHYSICS_COLLECTION)
    logger.info(
        f"OK: built collection '{PHYSICS_COLLECTION}' voi {len(docs)} few-shot examples."
    )


if __name__ == "__main__":
    main()
