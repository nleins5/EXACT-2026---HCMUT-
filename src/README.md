# src/

Source code chính của hệ thống EXACT-2026.

## Cấu trúc

```
src/
├── __init__.py
├── agent/          # LangGraph pipeline (graph, nodes, prompts, LLM)
├── core/           # Config loader (setting.yaml + .env)
├── process/        # (reserved — chưa dùng)
├── retrieval/      # RAG engine (hybrid search + reranker)
└── utils/          # Tiện ích dùng chung (logger, code extractor, Z3 parser)
```

## Luồng chính

```
Input (question + premises)
  → classifier (rule-based)
  → [logic branch]  formalizer → solver (Z3) → explanation
  → [physics branch] RAG → formalizer → solver (SymPy) → explanation
  → Output (ExactResponse JSON)
```

## Dependency

- `agent/` phụ thuộc `core/`, `utils/`, `retrieval/`
- `retrieval/` phụ thuộc `core/`
- `core/` phụ thuộc `utils/` (logger)
- `utils/` không phụ thuộc module nào khác trong `src/`
