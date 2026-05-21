# EXACT-2026: Agentic AI cho Logic & Vật lý

![EXACT-2026 Banner](https://img.shields.io/badge/Project-EXACT--2026-blueviolet?style=for-the-badge&logo=ai)
![LangGraph](https://img.shields.io/badge/Framework-LangGraph-orange?style=flat-square)
![LlamaIndex](https://img.shields.io/badge/Retrieval-LlamaIndex-blue?style=flat-square)
![Z3](https://img.shields.io/badge/Solver-Z3-green?style=flat-square)
![SymPy](https://img.shields.io/badge/Math-SymPy-red?style=flat-square)

**EXACT-2026** là hệ thống Agentic AI tiên tiến được xây dựng trên nền tảng LangGraph, chuyên giải quyết hai loại bài toán giáo dục:

- **Type 1**: Logic-Based Educational Queries (suy luận logic từ các giả thiết)
- **Type 2**: Physics Problems (giải bài toán vật lý định lượng)

Hệ thống kết hợp khả năng suy luận của LLM với các công cụ tính toán hình thức (Z3 cho logic, SymPy cho vật lý).

---

## 🚀 Tính năng chính

- **🧠 Dual-specialist LLM**: 2 mô hình fine-tuned chuyên biệt — `Qwen2.5-Coder-7B` sinh code Z3/SymPy, `Qwen2.5-7B-Instruct` sinh giải thích JSON.

- **🛰️ OpenAI-compatible serving**: Phục vụ qua `llama-server` (llama.cpp) — expose `/v1/chat/completions`.

- **♻️ Single-resident swap**: `LlamaServerSupervisor` quản lý vòng đời tiến trình, swap GGUF coder ↔ instruct giữa các stage.

- **⚡ Logic & Physics Solvers**: Code Z3 / SymPy chạy trong subprocess an toàn (timeout 30s).

- **🔍 Hybrid Retrieval cho Physics**: BM25 + Vector + reranker, truy xuất formulas + examples cho `physics_formalizer`.

- **🛡️ Branching success/error**: Khi solver fail, explanation node đọc code lỗi như hint và tự suy luận.

---

## 🏗️ Kiến trúc hệ thống

```
EXACT-2026/
├── bin/llama-cpp/              # llama-server.exe + DLLs
├── config/                     # setting.yaml + logging.yaml
├── data/
│   ├── collected/              # Dataset thu thập (electro textbook)
│   ├── distilled/              # Physics KB cho RAG (formulas)
│   ├── finetune/               # Dataset fine-tune (coder + instruct)
│   ├── external/               # PhysicsFormulae (GitHub)
│   └── EXACT2026_dataset_2026-05-15/  # Dataset BTC gốc
├── fine_tune/                  # Notebook Colab fine-tune
├── models/
│   ├── download_models.py      # Script tải GGUF từ HuggingFace
│   └── exact-2026/             # GGUF weights (Coder + Instruct)
├── scripts/
│   ├── data_prep/              # Tạo dataset fine-tune
│   ├── rag/                    # Build vector index
│   ├── convert_logic_to_z3.py  # Convert BTC Logic → Z3 code
│   └── convert_physics_to_sympy.py  # Convert BTC Physics → SymPy
├── src/
│   ├── agent/
│   │   ├── llm/                # ServerSupervisor + LLMFactory + OpenAI client
│   │   ├── nodes/              # 8 nodes: classifier, formalizer, solver, explanation
│   │   ├── prompts/            # Prompt templates
│   │   ├── graph.py            # LangGraph pipeline
│   │   ├── schema.py           # ExactResponse (Pydantic)
│   │   └── state.py            # AgentState
│   ├── core/                   # Config loader (Pydantic Settings)
│   ├── retrieval/              # Hybrid search + reranker
│   └── utils/                  # Logger, code_extract, z3_parser
├── storage/vector_db/          # LlamaIndex persistent index
├── test_pipeline.py            # Test end-to-end (5 bài mẫu)
└── requirements.txt
```

### Pipeline (1 request)

```
classify (rule-based)
  ├─ logic   → logic_formalizer (Coder) → logic_solver (Z3) → logic_explanation (Instruct)
  └─ physics → physics_rag → physics_formalizer (Coder) → physics_solver (SymPy) → physics_explanation (Instruct)
```

Mỗi request swap tối đa 1 lần (Coder → Instruct). Solver chạy subprocess, timeout 30s.

---

## 🛠️ Công nghệ sử dụng

| Layer | Công nghệ |
|-------|-----------|
| Orchestration | LangGraph + LangChain |
| LLM Serving | `llama-server` (llama.cpp), OpenAI-compatible |
| LLM Client | `langchain-openai` (ChatOpenAI) |
| Structured Output | Pydantic v2 + `with_structured_output()` |
| Retrieval | LlamaIndex (BM25 + Vector) + `BAAI/bge-reranker-base` |
| Embedding | `BAAI/bge-m3` (multilingual) |
| Solvers | `z3-solver`, `sympy` (subprocess) |
| Config | `pydantic-settings` + YAML |
| Tracing | LangSmith (optional) |

---

## 🏁 Bắt đầu

### 1. Cài đặt môi trường

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Tải model GGUF

```powershell
cd models
python download_models.py
```

Tải 2 file GGUF (~4.7GB mỗi file) từ `HoangKhangHCMUS/exact-2026`.

### 3. Cài đặt llama-server

Tải binary từ [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases), giải nén vào `bin/llama-cpp/`.

### 4. Build RAG index

```powershell
.\venv\Scripts\python.exe -m scripts.rag.build_physics_index
```

### 5. Test pipeline

```powershell
.\venv\Scripts\python.exe test_pipeline.py
```

### 6. Sử dụng trong code

```python
from src.agent.llm.factory import LLMFactory
from src.agent.llm.server_supervisor import LlamaServerSupervisor
from src.agent.graph import build_graph

supervisor = LlamaServerSupervisor()
LLMFactory.init(supervisor)
graph = build_graph()

result = graph.invoke({
    "question": "Calculate GPE of 5kg at 10m (g=9.8)",
    "premises": [],
    "task_type": "logic",
    "intermediate_answer": {...},
    "final_answer": {...},
})
```

---

## 📁 Cấu trúc thư mục

| Thư mục | Mục đích |
| ------- | -------- |
| `src/agent/` | Pipeline LangGraph (8 nodes) |
| `src/retrieval/` | Hybrid RAG (BM25 + vector + reranker) |
| `src/core/` | Config loader (setting.yaml → Pydantic) |
| `src/utils/` | Logger, code extractor, Z3 parser |
| `scripts/data_prep/` | Build fine-tune datasets |
| `scripts/rag/` | Build vector index |
| `data/` | Datasets (BTC, finetune, distilled, external) |
| `fine_tune/` | Notebook Colab fine-tune |
| `models/` | GGUF weights + download script |
| `bin/` | llama-server binary |
| `config/` | setting.yaml + logging.yaml |

---

## 📖 Tài liệu

- **`DATA_SOURCES_TEMP.md`** — Chi tiết nguồn dữ liệu
- **Mỗi folder** đều có `README.md` riêng

---

## 📝 Giấy phép

Dự án phục vụ cuộc thi **EXACT 2026** tại IEEE IJCNN 2026.
