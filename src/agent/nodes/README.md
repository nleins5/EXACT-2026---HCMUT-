# src/agent/nodes/

Các node trong LangGraph pipeline — mỗi file là 1 bước xử lý.

## Cấu trúc

```
nodes/
├── __init__.py
├── classifier.py           # Phân loại: logic vs physics (rule-based)
├── logic_formalizer.py     # Sinh code Z3 từ premises (Coder model)
├── logic_solver.py         # Chạy Z3 code, parse output
├── logic_explanation.py    # Sinh ExactResponse JSON (Instruct model)
├── physics_rag.py          # Truy vấn RAG lấy formula + examples
├── physics_formalizer.py   # Sinh code SymPy từ bài toán (Coder model)
├── physics_solver.py       # Chạy SymPy code, parse output
└── physics_explanation.py  # Sinh ExactResponse JSON (Instruct model)
```

## Pipeline flow

```
classifier
  ├── [logic]   → logic_formalizer → logic_solver → logic_explanation
  └── [physics] → physics_rag → physics_formalizer → physics_solver → physics_explanation
```

## Chi tiết

| Node | Model | Input | Output |
|------|-------|-------|--------|
| `classifier` | Rule-based | question, premises | `problem_type` |
| `logic_formalizer` | Coder | question, premises | Z3 code |
| `logic_solver` | subprocess | Z3 code | True/False/Unknown |
| `logic_explanation` | Instruct | solver output | ExactResponse |
| `physics_rag` | Embedding + Reranker | question | formulas + examples |
| `physics_formalizer` | Coder | question, RAG context | SymPy code |
| `physics_solver` | subprocess | SymPy code | numerical answer |
| `physics_explanation` | Instruct | solver output | ExactResponse |
