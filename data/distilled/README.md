# Distilled physics knowledge base

This directory is the input location for the optional physics RAG index.

The repository currently ships documentation only; it does **not** ship a
runtime JSONL corpus or persisted vector index. When no index exists,
`physics_rag_node` skips retrieval immediately so physics requests do not spend
their request budget downloading embedding or reranker models.

## Accepted record schema

`scripts/rag/build_physics_index.py` accepts disclosed, verified JSONL records:

```json
{
  "id": "example-1",
  "source": "disclosed-source",
  "problem": "R1=30 Ohm and R2=60 Ohm in parallel.",
  "topic": "electric_circuits",
  "formulas": ["R_eq = R1*R2/(R1+R2)"],
  "symbols": {"R1": "resistance 1", "R2": "resistance 2"},
  "sympy_code": "import sympy as sp\n...",
  "answer": "20 Ohm",
  "verified": true
}
```

Only records with `verified: true` are indexed. Records whose IDs start with
`QA` are excluded according to the official competition clarification.

## Build an index

```bash
python -m scripts.rag.build_physics_index \
  --input data/distilled/physics_kb.verified.jsonl \
  --rebuild
```

The input corpus must be generated using deterministic/open-source tooling and
fully disclosed before submission. Closed-source teacher APIs are prohibited.

Generated JSONL files and `storage/` are ignored by Git because they may be
large and reproducible. A deployment that intends to use RAG must build or
restore the index before starting the API.
