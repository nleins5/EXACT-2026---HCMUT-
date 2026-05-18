# EXACT 2026 — Kế hoạch triển khai LangGraph (tuân thủ BTC)

Mục tiêu: thiết kế đầy đủ kiến trúc agent EXACT 2026 sao cho:
- (a) Tuân thủ tất cả ràng buộc trong `QA.pdf` và `EXACT_Slides.pdf`.
- (b) Gán model rõ ràng cho từng node để sau này plug fine-tuned weights vào dễ.
- (c) Code chất lượng portfolio kể cả khi không đạt timing 60s.

Đây là **planning artifact** — chưa code gì cho tới khi bạn duyệt.

---

## 1. Audit compliance so với code hiện tại

| Quy tắc | Nguồn | Trạng thái hiện tại | Hành động |
|---|---|---|---|
| Model ≤ 8B-class | QA Q1 | ✅ DeepSeek-R1-0528-Qwen3-8B (8.19B, eligible) | giữ |
| MoE tính theo total params | QA Q2 | ✅ không dùng MoE | giữ |
| Tuần tự, single-resident | QA Q3 | ⚠️ pipeline tuần tự nhưng factory cache nhiều client cùng lúc | **fix**: swap process + GC giữa stage |
| **OpenAI-style serving (Q5)** | QA Q5 | ❌ đang dùng `ChatLlamaCpp` in-process | **fix bắt buộc**: dùng `llama-server` (C++ native từ llama.cpp) cho cả dev và prod |
| License open-source | QA Q4 | ✅ Qwen/DeepSeek license OK | giữ |
| Cho phép tools | QA Q7 | ✅ Z3, SymPy đã tích hợp | giữ |
| Cho phép RAG | QA Q8 | ✅ đã wired | giữ |
| Synthetic data từ closed LLM chỉ cho training | QA Q10 | n/a (chưa fine-tune) | document trong Data Disclosure |
| Data Disclosure Document | QA Q11 | ❌ chưa có | **fix**: scaffold `docs/DATA_DISCLOSURE.md` |
| Hard cap 60s/request | QA Q13 | ⚠️ đang enforce qua `asyncio.wait_for`, configurable | giữ; default 58s ở prod |
| Live API uptime | QA Q15 | ✅ middleware đã hỗ trợ | giữ |
| Single endpoint cho cả Type 1 & 2 | QA Q17, Q18 | ✅ `/predict` đã accept cả 2 dạng | giữ |
| Lọc sample `QA`-prefix trong Type 2 | QA Q19 | ❌ chưa làm | **fix**: thêm filter trong data loader |
| Notation toán | QA Q20 | n/a (eval-side normalize) | document |
| Scoring P1/P2/P3 | QA Q21 | một phần — schema đã đủ field | giữ; tinh chỉnh prompt để fill `cot/premises/confidence` |

> [!IMPORTANT]
> Lỗ hổng lớn nhất là **Q5 (serving)**. BTC sẽ hit `/v1/models` để verify model nào đang load.
> Lời văn Q5: *"vLLM (or compatible OpenAI-style serving framework)"*. **`llama-server`** từ
> repo `ggerganov/llama.cpp` đáp ứng đầy đủ điều kiện này, lại nhanh và nhẹ hơn vLLM trên CPU.

---

## 2. Gán model cho từng node

Slide 26 của `EXACT_Slides.pdf` đề xuất Type 1 → Z3, Type 2 → SymPy. Q3 cho phép swap model
giữa các stage miễn là tại mọi thời điểm chỉ 1 model resident.

### Stage map (1 request, cả 2 type chia sẻ pipeline này)

```
            ┌─ classify (rule-based, không LLM)
            │
            ├─ Type 1: logic_formalizer  ┐
            │                            ├─ chia sẻ: model code-generation
            └─ Type 2: physics_formalizer┘
                       ↓
                   solver (Z3 / SymPy, không LLM)
                       ↓
            ┌─ Type 1: logic_explanation ┐
            │                            ├─ chia sẻ: model NL/instruct
            └─ Type 2: physics_explanation┘
                       ↓
                  ExactResponse
```

### Đề xuất model

| Stage | Vai trò model | Base model gợi ý | Lý do |
|---|---|---|---|
| classify | không cần | n/a | đếm `premises` đủ — tiết kiệm 1 swap |
| logic_formalizer | code-instruct | `Qwen2.5-Coder-7B-Instruct` (~7.6B) | model code 8B-class tốt nhất, sinh Z3 syntax chuẩn |
| physics_formalizer | code-instruct | cùng trên | SymPy cũng là Python — tái dùng Coder, không swap |
| logic_explanation | NL-instruct | `Llama-3.1-8B-Instruct` hoặc `Qwen2.5-7B-Instruct` | NL trôi chảy, sinh cot/premises tốt |
| physics_explanation | NL-instruct | cùng trên | hiểu unit + reasoning |

### Tại sao chọn 2 model, không phải 3 hay 1

* **1 model**: phải compromise giữa code và prose. Test DeepSeek-R1 vừa rồi cho thấy hậu quả — Z3 syntax error mọi lần.
* **3 model** (Coder + LogicNL + PhysicsNL): mỗi request cần 2 swap. Không đáng; 1 Instruct lo NL tốt cho cả 2 nhánh.
* **2 model** (Coder + Instruct): đúng 1 swap/request (formalizer → explanation). Mỗi stage dùng model chuyên biệt cho mục đích đó.

> [!TIP]
> Nếu chọn `Qwen2.5-Coder-7B-Instruct` + `Qwen2.5-7B-Instruct`, cả 2 share Qwen tokenizer family.
> Tooling đánh giá downstream nhất quán, prompt template tái dùng được.

> [!IMPORTANT]
> Bạn đã nói sẽ fine-tune. Plug fine-tuned weights vào đúng 2 slot này.
> Architecture dưới coi `model_path` là placeholder.

---

## 3. Kiến trúc LangGraph (chốt)

### Nodes

```
__start__
  │
  ▼
classify  (rule-based; đọc trường premises)
  │
  ├──► logic_formalizer ──► logic_solver ──► logic_explanation ──► __end__
  │
  └──► physics_rag ──► physics_formalizer ──► physics_solver ──► physics_explanation ──► __end__
```

### Hợp đồng từng node

| Node | Input từ state | Output ra state | Model dùng |
|---|---|---|---|
| classify | `premises[]`, `question` | `task_type ∈ {logic, physics}` | không |
| physics_rag | `question` | `context: str` | không (FAISS) |
| logic_formalizer | `question`, `premises` | `intermediate.generated_code` | **Coder** |
| physics_formalizer | `question`, `context` | `intermediate.generated_code` | **Coder** |
| logic_solver | `intermediate.generated_code` | `intermediate.code_output`, `code_error`, `error_message` | không (subprocess Z3) |
| physics_solver | `intermediate.generated_code` | như trên | không (subprocess SymPy) |
| logic_explanation | `code_error`, `code_output` hoặc `error_message` + code | `final_answer` | **Instruct** |
| physics_explanation | như trên | `final_answer` | **Instruct** |

### State (giữ nguyên từ refactor lần trước)

```python
class AgentState(TypedDict):
    question:    str
    premises:    list[str]
    task_type:   Literal["logic", "physics"]
    context:     str                       # output RAG cho physics
    intermediate_answer: dict              # {generated_code, code_output, code_error, error_message, ...}
    final_answer:        dict              # field ExactResponse
    error:       str
    collection_name: str
```

### Bất biến single-resident (Q3)

Tại mọi yield point trong graph: **tối đa 1 process LLM resident**.

Cách làm: `LLMFactory.activate(role)` được gọi bởi formalizer + explanation.
Nó signal `LlamaServerSupervisor` kill process cũ, start process mới với model tương ứng,
chờ `/v1/models` ready rồi trả `ChatOpenAI` client trỏ về port đó. Xem section 5.

---

## 4. Migrate sang `llama-server` (QA Q5)

### Vì sao phải làm

> Q5: "All teams must serve their LLM via vLLM (or compatible OpenAI-style serving framework). The committee may query your /v1/models endpoint at any time."

`langchain_community.ChatLlamaCpp` chạy in-process, không expose `/v1/*` qua HTTP — fail Q5 ngay.

### Vì sao chọn `llama-server` thay vì vLLM hay `llama_cpp.server`

| Tiêu chí | `llama-server` (C++ native) | `llama_cpp.server` (Python) | `vLLM` |
|---|---|---|---|
| Q5 (`/v1/models`, `/v1/chat/completions`) | ✅ | ✅ | ✅ |
| Hardware | CPU + GPU (CUDA/Vulkan/Metal) | CPU + GPU | GPU only thực dụng |
| Cold start (CPU) | ~3-6s | ~6-10s | n/a |
| Cold start (GPU) | ~10-15s | n/a | ~30s |
| Continuous batching | ✅ | ❌ | ✅ |
| Đóng gói | single binary | pip package | pip package, deps nặng |
| Phù hợp dev CPU | ✅ tốt nhất | ✅ ổn | ❌ overkill |
| Phù hợp prod GPU | ✅ tốt | ⚠️ chậm hơn | ✅ tối ưu nhất |

**Kết luận**: dùng `llama-server` thống nhất cả dev và prod. Cold start nhanh là yếu tố quyết
định cho dual-model swap. vLLM chỉ thắng khi batch nhiều request — không phải workload cuộc thi này.

### Thiết kế 2 layer

* **LLM serving layer** (process #1, do `LlamaServerSupervisor` quản): `llama-server` binary,
  exposes `/v1/chat/completions`, `/v1/models` trên port cố định 8001.
* **Agent layer** (process #2): FastAPI app, gọi LLM qua `langchain-openai`'s `ChatOpenAI`
  trỏ về `http://127.0.0.1:8001/v1`.

### Single-resident (Q3): swap process

Chỉ chạy **1** instance `llama-server` tại một thời điểm. Khi swap:

```
state: coder server đang chạy port 8001
↓
agent gọi LLMFactory.activate("instruct")
↓
LlamaServerSupervisor.swap_to("instruct"):
  1. SIGTERM process coder
  2. wait exit (timeout 5s)
  3. spawn llama-server với model instruct, cùng port 8001
  4. poll GET /v1/models cho tới khi 200 (timeout 30s)
↓
trả ChatOpenAI client (base_url cố định)
```

Port cố định 8001 → agent không cần đổi `base_url` giữa các stage. ChatOpenAI client cache được.

### Chi phí latency

| Hardware | 1 swap |
|---|---|
| Local CPU (mmap warm) | ~3-6s |
| Cloud GPU (CUDA init) | ~10-15s |

Mỗi request 1 swap (formalizer → explanation). Trên GPU 60s budget: 15s swap + ~45s 2 LLM call. Khả thi.

> [!WARNING]
> Trên GPU cuộc thi, nếu inference mỗi LLM call >20s, sẽ vượt budget.
> Track ở Phase 3 fork (section 9) — fall back về single-model.

---

## 5. Plan thay đổi code (file by file)

### File mới

#### [NEW] [config/setting.yaml](file:///d:/Exact%202026/config/setting.yaml) (overwrite cái cũ)

```yaml
app:
  project_name: Exact 2026
  version: 1.0.0
  debug: true

# Endpoint LLM OpenAI-compatible. llama-server bind cố định port 8001.
# Mỗi role có model_path riêng; LlamaServerSupervisor swap process khi role thay đổi.
llm:
  server:
    binary:        "bin/llama-cpp/llama-server.exe"   # Windows; Linux: bin/llama-cpp/llama-server
    host:          "127.0.0.1"
    port:          8001
    base_url:      "http://127.0.0.1:8001/v1"
    api_key:       "not-needed"
    startup_timeout_s: 30
    shutdown_timeout_s: 5
    n_ctx:         4096
    n_gpu_layers:  0    # 0 = CPU. -1 = full GPU. Override bằng env LLAMA_NGL.
    extra_args:    []   # ví dụ: ["--flash-attn", "--mlock"]

  coder:
    model_name:  "qwen2.5-coder-7b-instruct"   # alias trong /v1/models
    model_path:  "models/qwen2.5-coder-7b-instruct.Q4_K_M.gguf"
    temperature: 0.0
    max_tokens:  1024

  instruct:
    model_name:  "qwen2.5-7b-instruct"
    model_path:  "models/qwen2.5-7b-instruct.Q4_K_M.gguf"
    temperature: 0.0
    max_tokens:  1024

embedding:
  model_name: BAAI/bge-m3

retrieval:
  threshold: 0.5
  top_k: 5

storage:
  data_dir: data
  vector_db: storage/vector_db
  collection_name: exact

api:
  request_budget_seconds: 600   # cuộc thi: set 58 qua env var EXACT_REQUEST_BUDGET_SECONDS

langsmith:
  project: Exact 2026
  endpoint: https://api.smith.langchain.com
```

#### [NEW] [src/llm/server_supervisor.py](file:///d:/Exact%202026/src/llm/server_supervisor.py)

Module quản lý lifecycle `llama-server`:

```python
class LlamaServerSupervisor:
    """Quản lý 1 process llama-server, bind port cố định.

    BTC Q3 cấm 2 LLM resident cùng lúc → swap_to() phải kill process cũ
    trước khi spawn cái mới. Q5 yêu cầu OpenAI-style → llama-server lo phần đó.
    """

    def __init__(self, settings: LLMServerConfig): ...

    def swap_to(self, role: Literal["coder", "instruct"]) -> None:
        """Đảm bảo server đang chạy với model của role tương ứng."""

    def is_alive(self) -> bool: ...
    def shutdown(self) -> None: ...
    def _spawn(self, model_path: Path, model_name: str): ...
    def _wait_ready(self, timeout: int): ...   # poll /v1/models
    def _kill(self, timeout: int): ...
```

Singleton, attached vào FastAPI lifespan.

#### [NEW] [src/llm/openai_client.py](file:///d:/Exact%202026/src/llm/openai_client.py)

Thay `LlamaCppClient`. Wrap `ChatOpenAI` từ `langchain-openai`. Vì port cố định nên chỉ
cần 1 `ChatOpenAI` instance. `get_structured_llm(schema)` dùng `with_structured_output`
qua method json_mode (tương thích `llama-server`).

#### [NEW] [scripts/serve_models.ps1](file:///d:/Exact%202026/scripts/serve_models.ps1)

Helper PowerShell — gọi `llama-server.exe` trực tiếp với role chỉ định, để test thủ công
ngoài pipeline. App production dùng `LlamaServerSupervisor` trong code thay vì script này.

#### [NEW] [scripts/install_llama_server.ps1](file:///d:/Exact%202026/scripts/install_llama_server.ps1)

Hướng dẫn (in URL + checksum) tải pre-built binary từ
`https://github.com/ggerganov/llama.cpp/releases/latest`. Không tự download —
chỉ in lệnh để user chạy. Lý do: file binary phụ thuộc CPU/GPU runtime của user.

#### [NEW] [src/agent/data_loader.py](file:///d:/Exact%202026/src/agent/data_loader.py)

Load Type 2 (Physics) CSV và **lọc bỏ row có id prefix `QA`** theo Q19.

#### [NEW] [docs/DATA_DISCLOSURE.md](file:///d:/Exact%202026/docs/DATA_DISCLOSURE.md)

Skeleton tuân thủ Q11.

### File sửa

#### [MODIFY] [src/core/config.py](file:///d:/Exact%202026/src/core/config.py)

Thay `LLMConfig` phẳng bằng:

```python
class LLMServerConfig(BaseModel):
    binary: str
    host: str = "127.0.0.1"
    port: int = 8001
    base_url: str
    api_key: str = "not-needed"
    startup_timeout_s: int = 30
    shutdown_timeout_s: int = 5
    n_ctx: int = 4096
    n_gpu_layers: int = 0
    extra_args: list[str] = []

class LLMRoleConfig(BaseModel):
    model_name: str
    model_path: str
    temperature: float = 0.0
    max_tokens: int = 1024

class LLMConfig(BaseModel):
    server:   LLMServerConfig
    coder:    LLMRoleConfig
    instruct: LLMRoleConfig
```

#### [MODIFY] [src/llm/factory.py](file:///d:/Exact%202026/src/llm/factory.py)

```python
class LLMFactory:
    """Single-resident LLM access. Dùng LlamaServerSupervisor swap process khi đổi role.
    Q3 BTC: tại mọi thời điểm chỉ 1 model resident.
    """
    _supervisor: LlamaServerSupervisor | None = None
    _client: ChatOpenAI | None = None
    _active_role: str | None = None

    @classmethod
    def init(cls, supervisor: LlamaServerSupervisor) -> None: ...

    @classmethod
    def activate(cls, role: Literal["coder", "instruct"]) -> ChatOpenAI:
        if cls._active_role != role:
            cls._supervisor.swap_to(role)
            cls._active_role = role
        if cls._client is None:
            cls._client = OpenAILLMClient.build(role)   # port cố định
        else:
            # update model_name nếu khác (cùng instance)
            cls._client.model_name = role_to_model_name(role)
        return cls._client
```

API legacy `purpose=...` giữ làm thin shim (`code/reasoning → coder`, `summary → instruct`).

#### [MODIFY] [src/agent/nodes/logic_formalizer.py](file:///d:/Exact%202026/src/agent/nodes/logic_formalizer.py), [physics_formalizer.py](file:///d:/Exact%202026/src/agent/nodes/physics_formalizer.py)

Gọi `LLMFactory.activate("coder")`.

#### [MODIFY] [src/agent/nodes/logic_explanation.py](file:///d:/Exact%202026/src/agent/nodes/logic_explanation.py), [physics_explanation.py](file:///d:/Exact%202026/src/agent/nodes/physics_explanation.py)

Gọi `LLMFactory.activate("instruct")`. Branch 2-prompt theo `code_error` giữ nguyên.

#### [MODIFY] [src/agent/nodes/classifier.py](file:///d:/Exact%202026/src/agent/nodes/classifier.py)

Bỏ LLM call thừa (route đã rule-based). Tiết kiệm 1 swap/request.

#### [MODIFY] [src/app.py](file:///d:/Exact%202026/src/app.py)

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    supervisor = LlamaServerSupervisor(settings.llm.server)
    LLMFactory.init(supervisor)
    # warm-up: chuẩn bị Instruct (role default cho explanation, có thể đổi)
    supervisor.swap_to("instruct")
    yield
    supervisor.shutdown()
```

Lifespan kill `llama-server` khi uvicorn dừng — không để zombie process.

#### [MODIFY] [src/api/routes.py](file:///d:/Exact%202026/src/api/routes.py)

`request_budget_seconds` đọc từ `settings.api.request_budget_seconds`.

#### [MODIFY] [requirements.txt](file:///d:/Exact%202026/requirements.txt)

Thêm `langchain-openai`, `httpx`. **Bỏ** `langchain-community` (không cần ChatLlamaCpp).
**Bỏ** `llama-cpp-python` (vì dùng binary thay vì Python wrapper).

### File xóa

#### [DELETE] [src/llm/provider/ollama_client.py](file:///d:/Exact%202026/src/llm/provider/ollama_client.py)

Thay bằng `openai_client.py` + `server_supervisor.py`.

---

## 6. Dataset & Disclosure (QA Q11, Q19)

### Filter Q19

```python
# src/agent/data_loader.py
def load_physics_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    bad = df["id"].str.startswith("QA")
    if bad.any():
        logger.info(f"Loại {bad.sum()} sample QA-prefix theo QA.pdf §Q19")
        df = df[~bad].reset_index(drop=True)
    return df
```

### Skeleton Data Disclosure

`docs/DATA_DISCLOSURE.md` gồm:

* Dataset chính thức: EXACT 2026 Type 1 (411 records, 808 questions); EXACT 2026 Type 2 (1,755 raw → 1,354 sau Q19).
* External dataset: TBD khi bắt đầu fine-tune.
* Synthetic data từ closed LLM: TBD (model gì, bao nhiêu sample, mục đích).
* Crawled / scraped data: TBD.
* RAG corpora: TBD (KB công thức vật lý nếu xây).

---

## 7. Boost P3 score (QA Q21)

* **`premises`**: prompt `*_explanation` echo lại premise đã dùng. Schema-validated.
* **`cot`**: list 3-5 step. Cap length.
* **`confidence`**: heuristic — 0.9 nếu `code_error=False`, 0.5 nếu True.
* **`fol`**: extract trực tiếp từ Z3 code (đã làm trong test slide-18).

Prompt-level changes, không đụng graph.

---

## 8. Plan verification

### Test tự động

* `tests/test_classifier.py` — unit test, không LLM.
* `tests/test_data_loader.py` — assert filter Q19 drop row `QA`.
* `tests/test_supervisor_swap.py` — start coder → swap instruct → assert đúng 1 process resident, `/v1/models` đổi tên.
* `tests/test_openai_client.py` — mock `httpx`, assert request shape khớp `/v1/chat/completions`.
* `scratch/test_z3_slide_example.py` (đã có) — extend assert `code_error=False` khi plug Coder thật.

### Verify thủ công

1. `pwsh scripts/install_llama_server.ps1` → in URL + cách giải nén.
2. `uvicorn src.app:app --workers 1`. Log show: supervisor spawn instruct, ready /v1/models.
3. `curl http://127.0.0.1:8001/v1/models` trả `qwen2.5-7b-instruct`.
4. POST slide-18 logic example → log show swap instruct → coder → swap về instruct, response đúng schema.
5. POST physics Type 2 → tương tự, path coder + instruct.

### Check compliance BTC

* `curl /v1/models` đúng tên model đã khai (Q5).
* `Get-Process llama-server` lúc idle → đúng 1 process (Q3).
* `docs/DATA_DISCLOSURE.md` được fill trước submission (Q11).
* Budget 60s `/predict` bật trong env prod (Q13).

---

## 9. Phase fork: nếu swap quá chậm

Nếu sau khi deploy thực tế, swap >15s/request là không chấp nhận được, fall back về
**1 model Instruct** (Qwen2.5-7B-Instruct hoặc fine-tune của bạn) cho cả formalizer
và explanation. Mất chính xác syntax của Coder, đổi lại không swap.
Branch 2-prompt (success vs error) đã handle case Z3 fail.

Fork này chỉ cần đổi `LLMFactory.activate("coder")` → `activate("instruct")` trong formalizer.
Không phải vẽ lại graph.

---

## Tóm tắt việc

Nếu bạn duyệt nguyên trạng, mình sẽ:
1. Reshape `setting.yaml` và `core/config.py` (`server` + `coder` + `instruct`).
2. Add `src/llm/server_supervisor.py`.
3. Add `src/llm/openai_client.py`.
4. Viết lại `LLMFactory` với `init()` + `activate()` + supervisor.
5. Update 4 node `*_formalizer` / `*_explanation`.
6. Bỏ LLM call trong `classifier.py`.
7. Add `scripts/install_llama_server.ps1` + `scripts/serve_models.ps1` (manual test).
8. Add `src/agent/data_loader.py` với filter Q19.
9. Add `docs/DATA_DISCLOSURE.md` skeleton.
10. Update `src/app.py` lifespan, `src/api/routes.py`, `requirements.txt`.
11. Delete `src/llm/provider/ollama_client.py`.
12. Update `README.md` với boot sequence mới.

Ước tính diff: ~700 dòng add, ~280 dòng delete.

---

## Open Questions

> [!IMPORTANT]
> **Q1**: Có muốn mình thêm bước **download GGUF tự động** vào `install_llama_server.ps1`
> không? Mặc định plan trên là user tự tải. Nếu có, cần khoảng ~9GB tải về (Coder + Instruct
> Q4_K_M).

> [!IMPORTANT]
> **Q2**: Có muốn `LlamaServerSupervisor` log stdout/stderr `llama-server` ra file
> `logs/llama-server-{role}.log` không? Hữu ích để debug khi swap fail. Mặc định mình sẽ
> bật vì cost thấp.

> [!TIP]
> **Q3** (optional): Có muốn mình điền sẵn Data Disclosure với info dataset chính thức
> EXACT 2026 (411 record Type 1, 1354 sau filter Type 2) để bạn chỉ cần thêm fine-tuning rows sau? Effort thấp, value cao cho submission.
