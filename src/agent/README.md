# src/agent/

Thư mục này chứa code runtime cho LangGraph pipeline của EXACT-2026.

## Cấu trúc

```
src/agent/
├── README.md               # File này
├── __init__.py             # Export: run_pipeline, get_graph, build_graph
├── graph.py                # Định nghĩa LangGraph workflow
├── state.py                # AgentState TypedDict
├── schema.py               # ExactResponse Pydantic model
│
├── llm/                    # Tầng phục vụ LLM
│   ├── __init__.py
│   ├── base.py             # BaseLLM ABC
│   ├── factory.py          # LLMFactory singleton (init/activate)
│   ├── openai_client.py    # OpenAILLMClient (ChatOpenAI wrapper)
│   ├── server_supervisor.py # LlamaServerSupervisor (spawn/kill/swap)
│   └── embedding.py        # EmbeddingFactory (BGE-M3)
│
├── nodes/                  # Các node xử lý trong pipeline
│   ├── __init__.py
│   ├── classifier.py       # Phân loại logic/physics + Router
│   ├── logic_formalizer.py # Dịch logic → mã Z3-Python
│   ├── logic_solver.py     # Thực thi Z3, set code_error flag
│   ├── logic_explanation.py # Tổng hợp kết quả (2 prompt: success/error)
│   ├── physics_rag.py      # Truy xuất công thức vật lý
│   ├── physics_formalizer.py # Dịch vật lý → mã SymPy
│   ├── physics_solver.py   # Thực thi SymPy, set code_error flag
│   └── physics_explanation.py # Tổng hợp kết quả (2 prompt: success/error)
│
└── prompts/                # Prompts cho từng node
    ├── __init__.py
    ├── logic_formalizer.py # Z3_SYSTEM_PROMPT, Z3_USER_TEMPLATE
    ├── logic_explanation.py # LOGIC_OUTPUT_PROMPT, LOGIC_OUTPUT_ERROR_PROMPT
    ├── physics_formalizer.py # PHYSICS_SYSTEM_PROMPT, PHYSICS_USER_TEMPLATE
    └── physics_explanation.py # PHYSICS_OUTPUT_PROMPT, PHYSICS_OUTPUT_ERROR_PROMPT
```

## Mục đích

Thư mục `src/agent/` chứa:
- **Pipeline LangGraph**: 8 nodes xử lý sequential (classify → formalizer → solver → explanation × 2 types)
- **Tầng LLM**: LlamaServerSupervisor, LLMFactory, OpenAILLMClient
- **Prompts**: Prompts cho từng node (formalizer + explanation)

## Chi tiết từng module

### graph.py

**Các hàm chính**:
- `build_graph()` — Xây dựng LangGraph với 8 nodes
- `run_pipeline(question, premises, collection_name)` — Chạy pipeline
- `get_graph()` — Trả về graph đã build

**Pipeline**:
```
classify → [logic_path | physics_path] → END
```

### state.py

**AgentState** - TypedDict chia sẻ giữa các node:

```python
class AgentState(TypedDict):
    question: str
    premises: list[str]
    collection_name: str
    task_type: Literal["logic", "physics"]
    context: str                    # RAG context (physics)
    intermediate_answer: IntermediateAnswer
    final_answer: FinalAnswer
    error: str
```

### schema.py

**ExactResponse** - Pydantic model cho API response:

```python
class ExactResponse(BaseModel):
    answer: str
    explanation: str
    fol: str | None
    cot: list[str] | None
    premises: list[str] | None
    confidence: float | None
```

### llm/

**Tầng phục vụ LLM**:

| File | Mục đích |
|------|----------|
| `base.py` | BaseLLM ABC |
| `factory.py` | LLMFactory singleton (init/activate) |
| `openai_client.py` | OpenAILLMClient (ChatOpenAI wrapper) |
| `server_supervisor.py` | LlamaServerSupervisor (spawn/kill/swap GGUF) |
| `embedding.py` | EmbeddingFactory (BGE-M3) |

**LlamaServerSupervisor**:
- `swap_to(role)` — Kill process cũ, spawn mới với GGUF đúng role
- Poll `/v1/models` để check readiness
- atexit cleanup

**LLMFactory**:
- Singleton pattern
- `activate(role)` — Swap process nếu cần, trả OpenAILLMClient
- Cache client theo role hiện tại

### nodes/

**8 nodes trong pipeline**:

| File | Mục đích |
|------|----------|
| `classifier.py` | Rule-based classification (có premises → logic) |
| `logic_formalizer.py` | LLM Coder → Z3 code |
| `logic_solver.py` | Subprocess Z3 (timeout 30s) |
| `logic_explanation.py` | LLM Instruct → JSON ExactResponse (2 prompts) |
| `physics_rag.py` | Hybrid retrieval (BM25 + vector) |
| `physics_formalizer.py` | LLM Coder → SymPy code |
| `physics_solver.py` | Subprocess SymPy (timeout 30s) |
| `physics_explanation.py` | LLM Instruct → JSON ExactResponse (2 prompts) |

### prompts/

**Prompts cho từng node**:

| File | Prompts |
|------|---------|
| `logic_formalizer.py` | `Z3_SYSTEM_PROMPT`, `Z3_USER_TEMPLATE` |
| `logic_explanation.py` | `LOGIC_OUTPUT_PROMPT`, `LOGIC_OUTPUT_ERROR_PROMPT` |
| `physics_formalizer.py` | `PHYSICS_SYSTEM_PROMPT`, `PHYSICS_USER_TEMPLATE` |
| `physics_explanation.py` | `PHYSICS_OUTPUT_PROMPT`, `PHYSICS_OUTPUT_ERROR_PROMPT` |

**2-prompt branching**: Khi solver fail, explanation node dùng ERROR prompt để đọc code lỗi như hint và tự suy luận (không regenerate code).

## Workflow

```
Client POST /predict
  → run_pipeline()
    → classify (rule-based)
      → logic_formalizer (Coder) → logic_solver (Z3) → logic_explanation (Instruct)
      → physics_rag → physics_formalizer (Coder) → physics_solver (SymPy) → physics_explanation (Instruct)
  → PredictResponse
```

**Timing**: 2 LLM calls + 1 swap = ~10-15s (fit trong budget 60s)

## Yêu cầu

```powershell
pip install -r requirements.txt
```

- `langgraph`, `langchain-openai`, `z3-solver`, `sympy`, `qdrant-client`, `llama-index`
