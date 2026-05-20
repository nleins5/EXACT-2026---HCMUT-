# EXACT-2026: Agentic AI cho Logic & Vật lý

![EXACT-2026 Banner](https://img.shields.io/badge/Project-EXACT--2026-blueviolet?style=for-the-badge&logo=ai)
![LangGraph](https://img.shields.io/badge/Framework-LangGraph-orange?style=flat-square)
![LlamaIndex](https://img.shields.io/badge/Retrieval-LlamaIndex-blue?style=flat-square)
![Z3](https://img.shields.io/badge/Solver-Z3-green?style=flat-square)
![SymPy](https://img.shields.io/badge/Math-SymPy-red?style=flat-square)

**EXACT-2026** là hệ thống AI đại lý (Agentic AI) tiên tiến được xây dựng trên nền tảng LangGraph, chuyên giải quyết hai loại bài toán giáo dục:

- **Type 1**: Logic-Based Educational Queries (suy luận logic từ các giả thiết)
- **Type 2**: Physics Problems (giải bài toán vật lý định lượng)

Hệ thống kết hợp khả năng suy luận của LLM với các công cụ tính toán hình thức (Z3 cho logic, SymPy cho vật lý).

---

## 🚀 Tính năng chính

- **🧠 Dual-specialist LLM**: 2 mô hình fine-tuned chuyên biệt — `Qwen2.5-Coder-7B` sinh code Z3/SymPy, `Qwen2.5-7B-Instruct` sinh giải thích JSON. Tách vai trò → mỗi model làm tốt nhiệm vụ riêng.
- **🛰️ OpenAI-compatible serving**: Phục vụ qua `llama-server` (llama.cpp) thay vì in-process — đáp ứng BTC Q5 (`/v1/models`, `/v1/chat/completions`).
- **♻️ Single-resident swap**: `LlamaServerSupervisor` quản lý vòng đời tiến trình, swap GGUF coder ↔ instruct giữa các stage → tuân thủ BTC Q3 (1 model resident tại một thời điểm).
- **⚡ Logic & Physics Solvers**: Code Z3 / SymPy chạy trong subprocess an toàn (timeout 30s), kết quả feed thẳng vào Instruct node để tổng hợp `ExactResponse`.
- **🔍 Hybrid Retrieval cho Physics**: BM25 + Qdrant + reranker `BAAI/bge-reranker-base`, truy xuất few-shot SymPy examples từ `coder.jsonl` → cấp context cho `physics_formalizer`.
- **🛡️ Branching success/error**: Khi solver fail, explanation node đọc code lỗi như hint và tự suy luận → đảm bảo có đáp án trong mọi tình huống.
- **🤖 Distillation Pipeline**: Teacher LLM (Gemini Flash Lite) trích xuất physics KB từ BTC + Electro → build vector index cho RAG. Chi phí ~0.18 USD cho toàn bộ 1,594 bài.

---

## 🏗️ Kiến trúc hệ thống

```
EXACT-2026/
├── bin/llama-cpp/                  # llama-server.exe (binary, tải tay)
├── models/                         # GGUF weights (Coder + Instruct)
├── src/
│   ├── agent/
│   │   ├── llm/                    # LlamaServerSupervisor + LLMFactory + OpenAI client
│   │   ├── nodes/                  # classifier, *_formalizer, *_solver, *_explanation
│   │   ├── prompts/                # 1 file prompt cho mỗi node (English content)
│   │   ├── graph.py                # LangGraph pipeline (sequential, no fan-out)
│   │   ├── schema.py               # ExactResponse Pydantic
│   │   └── state.py                # AgentState shared across nodes
│   ├── retrieval/                  # Hybrid (vector + BM25) + reranker
│   ├── api/                        # FastAPI route /predict
│   ├── core/                       # config (nested LLMConfig)
│   └── app.py                      # FastAPI lifespan: spawn supervisor
├── scripts/
│   ├── data_prep/                  # build coder.jsonl + instruct.jsonl
│   ├── rag/build_physics_index.py  # build Qdrant từ coder.jsonl
│   └── install_llama_server.ps1    # guide tải llama-server.exe
├── data/finetune/                  # 4 JSONL (đã filter Q19)
├── tests/                          # pytest unit (classifier, factory, prompts)
├── config/setting.yaml             # nested: server + coder + instruct
└── requirements.txt
```

### Pipeline (1 request)

```
classify (rule)              ← không LLM, dựa vào premises[]
  ├─ logic   → logic_formalizer (Coder)   → logic_solver (Z3)    → logic_explanation (Instruct)
  └─ physics → physics_rag (BM25+vector)  → physics_formalizer (Coder) → physics_solver (SymPy) → physics_explanation (Instruct)
```

Mỗi request swap đúng 1 lần (Coder → Instruct). Solver chạy subprocess Python, timeout 30s. Nếu solver fail, explanation node dùng prompt **ERROR branch** đọc code như hint.

---

## 🛠️ Công nghệ sử dụng

- **Orchestration**: LangGraph + LangChain (LangSmith tracing optional)
- **Serving LLM**: `llama-server` (llama.cpp C++ native), OpenAI-compatible `/v1/chat/completions`
- **LLM Client**: `langchain-openai` + `httpx`
- **Retrieval**: LlamaIndex hybrid (BM25 + Qdrant) + `BAAI/bge-reranker-base`
- **Embedding**: `BAAI/bge-m3` (multilingual)
- **Solvers**: `z3-solver`, `sympy` (chạy trong subprocess)
- **API**: FastAPI + Uvicorn
- **Validation**: Pydantic v2 + `pydantic-settings`

---

## 🏁 Bắt đầu

### 1. Cài đặt môi trường

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Cài đặt `llama-server` (binary)

```powershell
pwsh scripts/install_llama_server.ps1
```

Script sẽ in URL release của `ggerganov/llama.cpp` và hướng dẫn copy binary vào `bin/llama-cpp/llama-server.exe`. Chọn build phù hợp CPU/GPU của bạn (CPU x64 / CUDA 12 / Vulkan).

### 3. Tải GGUF weights

Đặt vào thư mục `models/`:

- `qwen2.5-coder-7b-instruct.Q4_K_M.gguf` (~4.6GB)
- `qwen2.5-7b-instruct.Q4_K_M.gguf` (~4.6GB)

Sau khi fine-tune xong, thay 2 file GGUF này bằng phiên bản đã fine-tune. Đường dẫn được khai báo trong `config/setting.yaml`.

### 4. (Tùy chọn) Build RAG index cho Physics

```powershell
python -m scripts.rag.build_physics_index
```

### 5. Chạy service

```powershell
uvicorn src.app:app --host 0.0.0.0 --port 8000 --workers 1
```

Kiểm tra:

```powershell
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8001/v1/models   # llama-server endpoint
```

### 6. Test pipeline

```powershell
.\venv\Scripts\python.exe -m pytest tests/ -v
```

---

## 📋 Tuân thủ BTC EXACT 2026

| Quy tắc                    | Cách tuân thủ                                                 |
| -------------------------- | ------------------------------------------------------------- |
| Model ≤ 8B                 | Qwen2.5 7B class                                              |
| Single-resident            | `LlamaServerSupervisor.swap_to()` kill cũ trước khi spawn mới |
| OpenAI-style serving       | `llama-server` expose `/v1/chat/completions` + `/v1/models`   |
| Cho phép tools             | Z3 + SymPy chạy subprocess                                    |
| Cho phép RAG               | Hybrid retriever cho physics_formalizer                       |
| Hard cap 60s/request       | `asyncio.wait_for` trong `routes.py`                          |
| Single endpoint Type 1 + 2 | `POST /predict`, classifier route theo `premises`             |
| Filter sample `QA-`        | Áp dụng tại `scripts/data_prep/`                              |

---

## 📁 Cấu trúc thư mục quan trọng

| Thư mục | Mục đích |
| ------- | -------- |
| `src/agent/` | Pipeline LangGraph (8 nodes: classifier, formalizer, solver, explanation × 2 types) |
| `src/api/` | FastAPI endpoints (`/predict`, `/health`) |
| `src/retrieval/` | Hybrid RAG (BM25 + vector + reranker) |
| `scripts/data_prep/` | Build fine-tune datasets (coder.jsonl, instruct.jsonl) |
| `scripts/distill/` | Build KB cho RAG (distill từ BTC + PhysicsFormulae) |
| `scripts/rag/` | Build Qdrant index cho physics |
| `data/finetune/` | ChatML datasets đã filter (sẵn sàng fine-tune) |
| `data/distilled/` | Physics knowledge base (formulas + examples) |
| `data/` | [Tất cả dữ liệu](data/README.md) (BTC, finetune, distilled, external) |
| `config/` | Cấu hình YAML + logging |

---

## 📖 Tài liệu chi tiết

Tài liệu đầy đủ bằng tiếng Việt nằm tại:

- **`Implementation/HUONG_DAN_TOAN_DIEN.md`** — Hướng dẫn toàn diện (1823 dòng) bao gồm:
  - Bối cảnh cuộc thi EXACT 2026
  - Kiến trúc hệ thống chi tiết
  - Pipeline LangGraph từng node
  - Hệ thống prompt engineering
  - RAG cho vật lý
  - Pipeline data prep & distillation
  - HTTP API & config
  - Tuân thủ BTC Q1–Q21
  - Hướng dẫn cài đặt & chạy
  - Trạng thái hiện tại & roadmap
  - Lỗi đã biết & cheat sheet

- **`data/README.md`** — Hướng dẫn chi tiết về toàn bộ dữ liệu:
  - Dataset BTC chính thức (Type 1 + Type 2)
  - Datasets fine-tune (coder.jsonl, instruct.jsonl)
  - Physics KB cho RAG (distillation pipeline)
  - External resources (PhysicsFormulae)
  - Nguồn dữ liệu: FOLIO, Physics, Electro

---

## 📝 Giấy phép

Dự án này được phát triển phục vụ cho cuộc thi/nghiên cứu **EXACT 2026** tại IEEE IJCNN 2026.

---
