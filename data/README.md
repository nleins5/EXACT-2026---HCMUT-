# EXACT-2026 Data Directory

Thư mục này chứa toàn bộ dữ liệu cho project EXACT-2026.

## Cấu trúc

```
data/
├── EXACT2026_dataset_2026-05-15/   # Dataset chính thức BTC (source gốc)
├── finetune/                       # Datasets đã preprocess (sẵn FT)
├── distilled/                      # Physics KB cho RAG
├── collected/                      # Dữ liệu thu thập thêm (electro)
├── external/                       # External resources (PhysicsFormulae)
├── EXACT_Slides.pdf                # Slide BTC chính thức
└── QA.pdf                          # Q&A BTC (Q1–Q21)
```

---

## 1. EXACT2026_dataset_2026-05-15/ — Dataset BTC chính thức

**Nguồn**: Cuộc thi EXACT 2026 tại IEEE IJCNN 2026

| Thư mục | Mô tả |
|--------|-------|
| `Logic_Based_Educational_Queries_Text_Only/` | 411 records / 808 câu hỏi Type 1 |
| `Physics_Problems_Text_Only/` | 1,352 bài Type 2 (đã filter `QA-`) |

**Scope vật lý**: chỉ `electric_circuits` + `electrostatics` (không cơ học, nhiệt, quang, lượng tử).

**Files**:
- `Logic_Based_Educational_Queries.json` — Type 1 với premises-NL + FOL
- `Physics_Problems_Text_Only.csv` — Type 2 với question + cot + answer + unit
- `CHANGELOG_TYPE1.md` / `CHANGELOG_TYPE2.md` — Lịch sử fix bug

**Ghi chú**: Đây là dataset gốc, **KHÔNG commit lên git** (file lớn). Chỉ commit changelog.

---

## 2. finetune/ — Datasets đã preprocess (sẵn fine-tune)

**Nguồn**: Xử lý từ `EXACT2026_dataset_2026-05-15/` + FOLIO + Electro

| File | Records | Mục đích |
|------|---------|----------|
| `coder.jsonl` | ~1,391 | Train Coder sinh code Z3/SymPy |
| `coder.eval.jsonl` | ~155 | Validation split (10%) |
| `instruct.jsonl` | ~2,518 | Train Instruct sinh JSON ExactResponse |
| `instruct.eval.jsonl` | ~280 | Validation split (10%) |
| `coder.STATS.md` / `instruct.STATS.md` | — | Báo cáo phân phối |

**Nguồn bổ sung**:
- **FOLIO** (yale-nlp/FOLIO): ~1,204 records premises-FOL → Z3 entailment
- **Electro**: 242 bài điện từ từ textbook (đã verify code)

**Workflow**:
```powershell
# Regenerate datasets
.\venv\Scripts\python.exe -m scripts.data_prep.prepare_coder_dataset
.\venv\Scripts\python.exe -m scripts.data_prep.prepare_instruct_dataset
```

**Fine-tune trên Colab**:
1. Zip `data/finetune/` → upload Drive
2. Notebook 1: Coder (Qwen2.5-Coder-7B-Instruct)
3. Notebook 2: Instruct (Qwen2.5-7B-Instruct)
4. Export GGUF Q4_K_M → drop vào `models/`

---

## 3. distilled/ — Physics KB cho RAG

**Nguồn**: BTC Physics + Electro + PhysicsFormulae GitHub

**Scope**: chỉ `electrostatics` + `electric_circuits`

| File | Tracked? | Mô tả |
|------|----------|-------|
| `physics_kb.from_pf.jsonl` | ✅ git | 22 formulas + 6 constants = 28 records (verified) |
| `physics_kb.raw.jsonl` | ❌ ignore | Output thô teacher LLM |
| `physics_kb.verified.jsonl` | ❌ ignore | Sau khi exec SymPy |
| `cost_log.jsonl` | ❌ ignore | Token + latency log |

**Pipeline**:
```powershell
# 1. Distill từ BTC + electro (Gemini Flash Lite)
$env:GOOGLE_API_KEY = "<your-key>"
.\venv\Scripts\python.exe -m scripts.distill.distill_physics --source all

# 2. Verify SymPy code
.\venv\Scripts\python.exe -m scripts.distill.verify_kb

# 3. Refresh từ PhysicsFormulae
.\venv\Scripts\python.exe -m scripts.distill.fetch_physics_formulae --include-constants

# 4. Build Qdrant index
.\venv\Scripts\python.exe -m scripts.rag.build_physics_index --rebuild
```

**Chi phí**: ~0.18 USD cho toàn bộ 1,594 bài (trả phí), hoặc 0 USD qua đêm (free tier 15 RPM).

**Schema**: `KBRecord` với fields: `id`, `source`, `problem`, `topic`, `formulas`, `symbols`, `sympy_code`, `answer`, `derivation`, `verified`, `exec_output`, `exec_error`, `teacher_model`, `input_tokens`, `output_tokens`

---

## 4. collected/ — Dữ liệu thu thập thêm

| File | Records | Mô tả |
|------|---------|-------|
| `electro_dataset.jsonl` | 242 | Bài điện từ (raw) |
| `electro_sympy_dataset.jsonl` | 242 | Bài điện từ + SymPy code đã verify |

**Nguồn**: Textbook điện từ (không từ BTC).

---

## 5. external/ — External resources

| File | Mô tả |
|------|-------|
| `PhysicsFormulae_Compiled.json` | ~655KB cache từ [BenjaminTMilnes/PhysicsFormulae](https://github.com/BenjaminTMilnes/PhysicsFormulae) |

**Filter**: giữ `Classical Electromagnetism` + `Electric Circuits`, bỏ mechanics/thermodynamics/optics/quantum.

**Constants**: `epsilon_0`, `mu_0`, `e`, `k_e`, `m_e`, `m_p`

**License note**: Repo source không có LICENSE → facts (công thức) không bản quyền. LaTeX đã transform sang plain math.

---

## 6. EXACT_Slides.pdf & QA.pdf

- `EXACT_Slides.pdf`: Slide BTC chính thức (scope, format, rules)
- `QA.pdf`: Q&A BTC (Q1–Q21) — quy tắc kỹ thuật quan trọng

---

## Nguồn dataset gốc

| Dataset | Link | Mô tả |
|---------|------|-------|
| **Physics** | https://github.com/yale-nlp/Physics/tree/main/PHYSICS | Physics problems (FOLIO) |
| **FOLIO** | https://huggingface.co/datasets/yale-nlp/P-FOLIO | Formal logic problems |

---

## Quy tắc commit

- ✅ **Commit**: changelog, `physics_kb.from_pf.jsonl`, `README.md`
- ❌ **Ignore**: dataset lớn, output regen được, logs, models, storage

Xem `.gitignore` chi tiết.

---
