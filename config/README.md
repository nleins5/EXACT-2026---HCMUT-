# config/

File cấu hình cho toàn bộ hệ thống.

## Cấu trúc

```
config/
├── setting.yaml    # Cấu hình chính (LLM, RAG, storage, etc.)
└── logging.yaml    # Cấu hình logging (format, handlers, levels)
```

## setting.yaml

Cấu hình chính, được load bởi `src/core/config.py` → singleton `settings`.

### Các section

| Section | Mô tả |
|---------|--------|
| `app` | Tên project, version, debug |
| `llm.server` | Binary path, host, port, ctx_size, gpu_layers |
| `llm.coder` | Model path + params cho Coder (sinh code) |
| `llm.instruct` | Model path + params cho Instruct (sinh explanation) |
| `api` | Request budget timeout |
| `rag` | Reranker model (BAAI/bge-reranker-base) |
| `embedding` | Embedding model (BAAI/bge-m3) |
| `retrieval` | Threshold, top_k |
| `storage` | Data dir, vector DB path, collection name |
| `langsmith` | LangSmith tracing project |
| `distillation` | Offline pipeline config (teacher LLM, paths) |

### Override bằng env var

- `LANGSMITH_API_KEY`: enable LangSmith tracing.
- `GOOGLE_API_KEY`: cho distillation pipeline (Gemini).

## logging.yaml

- Logger `"exact"`: level INFO, output ra console + file `logs/exact.log`.
- Được load tự động khi import `src.utils.logger`.
