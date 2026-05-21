# scripts/rag/

Script xây dựng vector index cho RAG (Retrieval-Augmented Generation).

## Cấu trúc

```
rag/
├── __init__.py
└── build_physics_index.py   # Tạo LlamaIndex vector store từ distilled data
```

## build_physics_index.py

- **Input**: `data/distilled/physics_kb.formulas.jsonl` + `physics_kb.from_pf.jsonl`
- **Output**: `storage/vector_db/` (LlamaIndex persistent index)
- **Embedding**: BAAI/bge-m3 (multilingual, dense + sparse)
- **Chức năng**:
  1. Đọc JSONL formulas → tạo Document nodes.
  2. Embed bằng BGE-M3.
  3. Lưu vào VectorStoreIndex (persistent storage).

## Cách chạy

```bash
cd "Exact 2026"
venv\Scripts\python.exe -m scripts.rag.build_physics_index
```

## Khi nào cần chạy lại

- Khi thêm/sửa file trong `data/distilled/`.
- Khi đổi embedding model.
- Khi xóa `storage/vector_db/`.
