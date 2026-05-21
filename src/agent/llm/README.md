# src/agent/llm/

Tầng LLM client — quản lý process `llama-server` và giao tiếp qua OpenAI API.

## Cấu trúc

```
llm/
├── __init__.py
├── base.py              # Abstract base class (BaseLLM)
├── embedding.py         # Embedding model wrapper
├── factory.py           # LLMFactory singleton (activate coder/instruct)
├── openai_client.py     # OpenAI-compatible client → llama-server
└── server_supervisor.py # Quản lý process llama-server (spawn/kill/swap)
```

## Luồng hoạt động

```
LLMFactory.activate("coder")
  → ServerSupervisor.swap_to("coder")   # kill instruct, spawn coder
  → OpenAILLMClient(role="coder")       # tạo ChatOpenAI client
  → client.get_llm() / get_structured_llm(schema)
```

## Chi tiết

### factory.py

- `LLMFactory`: Singleton gateway. Gọi `init(supervisor)` 1 lần, sau đó `activate(role)` trong mỗi node.
- Tại 1 thời điểm chỉ 1 model resident (BTC Q3 constraint).

### server_supervisor.py

- `LlamaServerSupervisor`: spawn/kill `llama-server.exe` process.
- `swap_to(role)`: kill process cũ → spawn mới với GGUF tương ứng.
- Health check: poll `/v1/models` cho đến khi server ready.

### openai_client.py

- `OpenAILLMClient`: wrap `ChatOpenAI` (LangChain) trỏ về `llama-server` local.
- `get_llm()`: trả ChatOpenAI instance.
- `get_structured_llm(schema)`: dùng `with_structured_output(schema)` — Pydantic enforce.

### base.py

- `BaseLLM`: abstract interface (`get_llm`, `get_structured_llm`).

### embedding.py

- Wrapper cho embedding model (BGE-M3), dùng trong RAG.
