# data/

Thư mục này chứa toàn bộ dữ liệu cho project EXACT-2026.

## Cấu trúc

```
data/
├── README.md                           # File này
├── EXACT2026_dataset_2026-05-15/     # Dataset chính thức BTC (source gốc)
├── finetune/                           # Datasets đã preprocess (sẵn FT)
├── distilled/                          # Physics KB cho RAG
├── collected/                          # Dữ liệu thu thập thêm (electro)
├── external/                           # External resources (PhysicsFormulae)
├── EXACT_Slides.pdf                    # Slide BTC chính thức
└── QA.pdf                              # Q&A BTC (Q1–Q21)
```

## Mục đích

Thư mục `data/` chứa:
- **Dataset gốc**: BTC dataset (Type 1 + Type 2)
- **Dataset đã preprocess**: ChatML format cho fine-tune
- **Physics KB**: Knowledge base cho RAG
- **External resources**: PhysicsFormulae, Electro dataset

## Cấu trúc chi tiết

### EXACT2026_dataset_2026-05-15/

Dataset chính thức từ BTC (source gốc).

| Thư mục | Mô tả |
|--------|-------|
| `Logic_Based_Educational_Queries_Text_Only/` | 411 records / 808 câu hỏi Type 1 |
| `Physics_Problems_Text_Only/` | 1,352 bài Type 2 (đã filter `QA-`) |

**Files**:
- `Logic_Based_Educational_Queries.json` — Type 1 với premises-NL + FOL
- `Physics_Problems_Text_Only.csv` — Type 2 với question + cot + answer + unit
- `CHANGELOG_TYPE1.md` / `CHANGELOG_TYPE2.md` — Lịch sử fix bug

**Ghi chú**: Đây là dataset gốc, **KHÔNG commit lên git** (file lớn).

### finetune/

Datasets đã preprocess (ChatML format) cho fine-tune.

| File | Records | Mục đích |
|------|---------|----------|
| `coder.jsonl` | ~1,391 | Train Coder sinh code Z3/SymPy |
| `coder.eval.jsonl` | ~155 | Validation split (10%) |
| `instruct.jsonl` | ~2,518 | Train Instruct sinh JSON ExactResponse |
| `instruct.eval.jsonl` | ~280 | Validation split (10%) |

**Nguồn**: BTC Physics + FOLIO + Physics + Electro

**Cách sinh**:
```powershell
.\venv\Scripts\python.exe -m scripts.data_prep.prepare_coder_dataset
.\venv\Scripts\python.exe -m scripts.data_prep.prepare_instruct_dataset
```

**Fine-tune**: Xem `fine_tune/README.md`

### distilled/

Physics KB cho RAG (knowledge base).

| File | Tracked? | Mô tả |
|------|----------|-------|
| `physics_kb.from_pf.jsonl` | ✅ git | 22 formulas + 6 constants = 28 records |
| `physics_kb.raw.jsonl` | ❌ ignore | Output thô teacher LLM |
| `physics_kb.verified.jsonl` | ❌ ignore | Sau khi exec SymPy |
| `cost_log.jsonl` | ❌ ignore | Token + latency log |

**Pipeline**:
```powershell
.\venv\Scripts\python.exe -m scripts.distill.distill_physics --source all
.\venv\Scripts\python.exe -m scripts.distill.verify_kb
.\venv\Scripts\python.exe -m scripts.distill.fetch_physics_formulae --include-constants
.\venv\Scripts\python.exe -m scripts.rag.build_physics_index --rebuild
```

**Xem thêm**: `distilled/README.md`

### collected/

Dữ liệu thu thập thêm (textbook điện từ).

| File | Records | Mô tả |
|------|---------|-------|
| `electro_dataset.jsonl` | 242 | Bài điện từ (raw) |
| `electro_sympy_dataset.jsonl` | 242 | Bài điện từ + SymPy code đã verify |

**Ghi chú**: Đây là dataset **nội bộ** (không từ BTC), do bạn thu thập thêm.

### external/

External resources.

| File | Mô tả |
|------|-------|
| `PhysicsFormulae_Compiled.json` | Cache ~655KB từ PhysicsFormulae GitHub |

**Cách dùng**: File này được download bởi `scripts/distill/fetch_physics_formulae.py`.

### EXACT_Slides.pdf & QA.pdf

- `EXACT_Slides.pdf`: Slide BTC chính thức (scope, format, rules)
- `QA.pdf`: Q&A BTC (Q1–Q21) — quy tắc kỹ thuật quan trọng

## Quy tắc commit

- ✅ **Commit**: changelog, `physics_kb.from_pf.jsonl`, `README.md`
- ❌ **Ignore**: dataset lớn, output regen được, logs, models, storage

Xem `.gitignore` chi tiết.
