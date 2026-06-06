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
└── rag/                            # Build vector index
    ├── __init__.py
    └── build_physics_index.py      # Verified JSONL → 2 collection Qdrant
```

## Mục đích

Thư mục `scripts/` chứa:
- **Engine chuyển đổi**: FOL → Z3, LaTeX → SymPy
- **Pipeline data prep**: Build dataset fine-tune (coder.jsonl, instruct.jsonl)
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

### rag/

Build vector index Qdrant từ một JSONL corpus đã verified và fully disclosed.

| File | Mục đích |
|------|----------|
| `build_physics_index.py` | Build 2 collection: `physics_examples` + `physics_formulas` |
**Cách dùng**:
```powershell
.\venv\Scripts\python.exe -m scripts.rag.build_physics_index --input data/distilled/physics_kb.verified.jsonl --rebuild
```

## Workflow tổng

```
BTC Data → data_prep → coder.jsonl/instruct.jsonl → fine_tune → GGUF
Disclosed verified JSONL → rag → Qdrant index → runtime RAG
```

## Yêu cầu

```powershell
pip install -r requirements.txt
```

- **data_prep**: `datasets`, `z3-solver`, `sympy`, `pandas`, `python-dotenv`
- **rag**: `qdrant-client`, `llama-index`, `FlagEmbedding`
