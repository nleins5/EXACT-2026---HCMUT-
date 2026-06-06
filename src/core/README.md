# src/core/

Config loader — đọc `config/setting.yaml` + `.env`, expose singleton `settings`.

## Cấu trúc

```
core/
├── __init__.py
└── config.py       # Pydantic Settings, load YAML + env override
```

## config.py

- Dùng `pydantic-settings` (BaseSettings) + `pydantic` (BaseModel) cho từng section.
- Load file `config/setting.yaml` tại module level → export `settings` singleton.
- Hỗ trợ override bằng env var (ví dụ: `LANGSMITH_API_KEY`).

### Các section chính

| Section | Class | Mô tả |
|---------|-------|--------|
| `app` | `AppConfig` | Tên project, version, debug flag |
| `llm` | `LLMConfig` | Server binary + 2 role (coder, instruct) |
| `embedding` | `EmbeddingConfig` | Model embedding (BGE-M3) |
| `rag` | `RagConfig` | Reranker model |
| `retrieval` | `RetrievalConfig` | Threshold, top_k |
| `storage` | `StorageConfig` | Đường dẫn vector DB |
| `langsmith` | `LangsmithConfig` | Tracing config |
| `distillation` | `DistillationConfig` | Offline extraction config; closed-source LLM providers disabled |

## Cách dùng

```python
from src.core.config import settings

print(settings.llm.coder.model_path)
print(settings.retrieval.top_k)
```
