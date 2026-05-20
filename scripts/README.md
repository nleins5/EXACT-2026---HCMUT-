# scripts/

Thư mục này chứa các script Python phục vụ data pipeline, distillation và RAG cho EXACT-2026.

## Cấu trúc

```
scripts/
├── README.md                       # File này
├── convert_logic_to_z3.py          # Engine: FOL → Z3 Python code
├── convert_physics_to_sympy.py     # Engine: LaTeX → SymPy Python code
├── install_llama_server.ps1        # Hướng dẫn tải llama-server.exe
├── data_prep/                      # Pipeline build dataset fine-tune
│   ├── __init__.py
│   ├── _common.py                  # Loaders + ChatML formatter + verify exec
│   ├── prepare_coder_dataset.py    # → data/finetune/coder.jsonl
│   └── prepare_instruct_dataset.py # → data/finetune/instruct.jsonl
├── distill/                        # Knowledge distillation cho RAG
│   ├── __init__.py
│   ├── distill_physics.py          # Teacher LLM → physics_kb.raw.jsonl
│   ├── verify_kb.py                # Exec SymPy → mark verified
│   └── fetch_physics_formulae.py   # Pull công thức từ PhysicsFormulae GitHub
└── rag/                            # Build vector index + smoke test
    ├── __init__.py
    ├── build_physics_index.py      # JSONL → 2 collection Qdrant
    └── smoke_rag.py                # Test 3 query (Coulomb / out-scope / RLC)
```

## Mục đích

Thư mục `scripts/` chứa:
- **Engine chuyển đổi**: FOL → Z3, LaTeX → SymPy
- **Pipeline data prep**: Build dataset fine-tune (coder.jsonl, instruct.jsonl)
- **Pipeline distillation**: Build Physics KB cho RAG
- **Pipeline RAG**: Build vector index Qdrant

## Chi tiết từng module

### convert_logic_to_z3.py

**Mục đích**: Chuyển premises/conclusion từ First-Order Logic (Unicode) thành Python script dùng `z3-solver`.

**Conversion patterns**:
- `∀x P(x)` → `ForAll(x, P(x))`
- `∃x P(x)` → `Exists(x, P(x))`
- `¬P(x)` → `Not(P(x))`
- `P → Q` → `Implies(P, Q)`
- `P ∧ Q` → `And(P, Q)`
- `P ∨ Q` → `Or(P, Q)`

### convert_physics_to_sympy.py

**Mục đích**: Chuyển biểu thức LaTeX thành Python expression dùng `sympy`.

**Conversion patterns**:
- `\frac{a}{b}` → `((a)/(b))`
- `\sqrt{x}` → `sqrt(x)`
- `x^{2}` → `x**(2)`
- `\sin x` → `sin(x)`
- `\pi` → `pi`

### data_prep/

Pipeline build dataset fine-tune.

| File | Mục đích |
|------|----------|
| `_common.py` | Loaders (BTC, FOLIO, Electro), ChatML formatter, `verify_python()` |
| `prepare_coder_dataset.py` | Build `coder.jsonl` (FOL → Z3, LaTeX → SymPy) |
| `prepare_instruct_dataset.py` | Build `instruct.jsonl` (problem + code + output → JSON) |

**Output**:
- `data/finetune/coder.jsonl` (~1,391 records)
- `data/finetune/instruct.jsonl` (~2,518 records)

**Cách dùng**:
```powershell
.\venv\Scripts\python.exe -m scripts.data_prep.prepare_coder_dataset
.\venv\Scripts\python.exe -m scripts.data_prep.prepare_instruct_dataset
```

### distill/

Knowledge distillation cho RAG.

| File | Mục đích |
|------|----------|
| `distill_physics.py` | Teacher LLM (Gemini) trích xuất KB từ BTC/Electro |
| `verify_kb.py` | Exec SymPy code, mark `verified=true/false` |
| `fetch_physics_formulae.py` | Crawl PhysicsFormulae GitHub |

**Output**:
- `data/distilled/physics_kb.raw.jsonl`
- `data/distilled/physics_kb.verified.jsonl`
- `data/distilled/physics_kb.from_pf.jsonl` (28 records, commit lên git)

**Cách dùng**:
```powershell
.\venv\Scripts\python.exe -m scripts.distill.distill_physics --source all
.\venv\Scripts\python.exe -m scripts.distill.verify_kb
.\venv\Scripts\python.exe -m scripts.distill.fetch_physics_formulae --include-constants
```

### rag/

Build vector index Qdrant.

| File | Mục đích |
|------|----------|
| `build_physics_index.py` | Build 2 collection: `physics_examples` + `physics_formulas` |
| `smoke_rag.py` | Test retrieval với 3 query |

**Cách dùng**:
```powershell
.\venv\Scripts\python.exe -m scripts.rag.build_physics_index --rebuild
.\venv\Scripts\python.exe -m scripts.rag.smoke_rag
```

## Workflow tổng

```
BTC Data → data_prep → coder.jsonl/instruct.jsonl → fine_tune → GGUF
BTC/Electro → distill → physics_kb → rag → Qdrant index → runtime RAG
PhysicsFormulae → fetch_physics_formulae → physics_kb
```

## Yêu cầu

```powershell
pip install -r requirements.txt
```

- **data_prep**: `datasets`, `z3-solver`, `sympy`, `pandas`, `python-dotenv`
- **distill**: `google-generativeai`
- **rag**: `qdrant-client`, `llama-index`, `FlagEmbedding`
