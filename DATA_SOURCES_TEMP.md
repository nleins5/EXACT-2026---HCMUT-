# EXACT-2026 Data Sources

Tài liệu mô tả chi tiết từng dataset trong project, bao gồm nguồn gốc, mục đích, và script tạo ra.

---

## 1. DATASET CHO FINE-TUNE

### 1.1. Coder Dataset (Qwen2.5-Coder-7B-Instruct)

**Mục đích**: Train model sinh code Z3 (logic) hoặc SymPy (vật lý).

**Files**:
- `data/finetune/coder.jsonl` (~1,391 records)
- `data/finetune/coder.eval.jsonl` (~155 records — 10% holdout)

**Nguồn dữ liệu**:

| Source | Records | Mô tả | File code tạo |
|--------|---------|-------|---------------|
| **BTC Physics** | ~1,022 | Dataset BTC chính thức — Physics CSV → SymPy | `scripts/data_prep/prepare_coder_dataset.py` |
| **FOLIO** | ~177 | yale-nlp/FOLIO — premises-FOL → Z3 entailment | `scripts/data_prep/prepare_coder_dataset.py` |
| **Electro** | ~192 | Textbook điện từ → SymPy code | `scripts/data_prep/prepare_coder_dataset.py` |

**Chạy**:
```powershell
.\venv\Scripts\python.exe -m scripts.data_prep.prepare_coder_dataset
```

---

### 1.2. Instruct Dataset (Qwen2.5-7B-Instruct)

**Mục đích**: Train model sinh JSON `ExactResponse` từ (problem + code + code_output).

**Files**:
- `data/finetune/instruct.jsonl` (~2,518 records)
- `data/finetune/instruct.eval.jsonl` (~280 records — 10% holdout)

**Nguồn dữ liệu**:

| Source | Records | Mô tả | File code tạo |
|--------|---------|-------|---------------|
| **BTC Physics** | ~1,213 | Physics + SymPy code + CoT | `scripts/data_prep/prepare_instruct_dataset.py` |
| **FOLIO** | ~1,082 | Premises + Z3 code + label | `scripts/data_prep/prepare_instruct_dataset.py` |
| **Electro** | ~223 | Textbook điện từ + SymPy code | `scripts/data_prep/prepare_instruct_dataset.py` |

**Chạy**:
```powershell
.\venv\Scripts\python.exe -m scripts.data_prep.prepare_instruct_dataset
```

---

## 2. DATASET CHO RAG (Physics Knowledge Base)

**Mục đích**: Cung cấp công thức vật lý và ví dụ mẫu cho `physics_rag_node` tại runtime.

**Files trong `data/distilled/`**:

| File | Records | Nguồn | Script tạo |
|------|---------|-------|------------|
| `physics_kb.formulas.jsonl` | 363 | BTC Physics CSV (deterministic extraction, no closed-source LLM) | `scripts/convert_physics_to_sympy.py` |
| `physics_kb.from_pf.jsonl` | 29 | PhysicsFormulae GitHub (curated) | `scripts/convert_physics_to_sympy.py` |

**Build vector index**:
```powershell
.\venv\Scripts\python.exe -m scripts.rag.build_physics_index
```

**Output**: `storage/vector_db/` (LlamaIndex persistent index, dùng bởi `physics_rag_node`).

---

## 3. DATASET GỐC (BTC)

### 3.1. Logic-Based Educational Queries (Type 1)

**File**: `data/EXACT2026_dataset_2026-05-15/Logic_Based_Educational_Queries_Text_Only/Logic_Based_Educational_Queries.json`

**Records**: 411 records / 808 câu hỏi

**Fields**: `premises-NL`, `premises-FOL`, `question`, `answer`, `explanation`

**Mục đích**: Nguồn cho fine-tune Coder (Z3) và Instruct (logic explanation)

---

### 3.2. Physics Problems (Type 2)

**File**: `data/EXACT2026_dataset_2026-05-15/Physics_Problems_Text_Only/Physics_Problems_Text_Only.csv`

**Records**: 1,352 bài (đã filter `QA-`)

**Fields**: `id`, `question`, `cot`, `answer`, `unit`

**Mục đích**: Nguồn cho fine-tune Coder (SymPy), Instruct (physics explanation), và RAG (formula/worked-example extraction)

---

## 4. DATASET THU THẬP (Collected)

### 4.1. Electro Dataset

**File**: `data/collected/electro_dataset.jsonl` (242 records)

**Nguồn**: Textbook điện từ (thu thập thủ công)

**Fields**: `id`, `questions`, `solutions`, `final_answers`, `graphs`

**Mục đích**: Bổ sung cho fine-tune Coder + Instruct (physics)

### 4.2. Electro Sympy Dataset

**File**: `data/collected/electro_sympy_dataset.jsonl` (242 records)

**Nguồn**: Sinh từ `electro_dataset.jsonl` bằng script chuyển đổi SymPy/open-source symbolic tooling; không dùng closed-source LLM

**Fields**: `id`, `questions`, `solution`, `final_answers`, `sympy_code`

**Mục đích**: Code SymPy mẫu cho fine-tune Coder

---

## 5. EXTERNAL RESOURCES

### 5.1. PhysicsFormulae

**File**: `data/external/PhysicsFormulae_Compiled.json` (~655KB)

**Nguồn**: [BenjaminTMilnes/PhysicsFormulae](https://github.com/BenjaminTMilnes/PhysicsFormulae)

**Mục đích**: Công thức vật lý chuẩn → curated thành `physics_kb.from_pf.jsonl` cho RAG

---

## 6. TỔNG HỢP

| Dataset | Mục đích | Nguồn | Script | Records |
|---------|----------|-------|--------|---------|
| `coder.jsonl` | Fine-tune Coder | BTC + FOLIO + Electro | `scripts/data_prep/prepare_coder_dataset.py` | ~1,391 |
| `instruct.jsonl` | Fine-tune Instruct | BTC + FOLIO + Electro | `scripts/data_prep/prepare_instruct_dataset.py` | ~2,518 |
| `physics_kb.formulas.jsonl` | RAG (formulas) | BTC Physics (deterministic extraction) | `scripts/convert_physics_to_sympy.py` | 363 |
| `physics_kb.from_pf.jsonl` | RAG (formulas) | PhysicsFormulae GitHub | `scripts/convert_physics_to_sympy.py` | 29 |

---

## 7. FLOW DIAGRAM

```
BTC Logic JSON + FOLIO
    ↓
scripts/data_prep/prepare_coder_dataset.py (Z3 portion)
scripts/data_prep/prepare_instruct_dataset.py (explanation portion)
    ↓
data/finetune/coder.jsonl + instruct.jsonl → Fine-tune → GGUF

BTC Physics CSV + Electro
    ↓
scripts/data_prep/prepare_coder_dataset.py (SymPy portion)
scripts/data_prep/prepare_instruct_dataset.py (explanation portion)
    ↓
data/finetune/coder.jsonl + instruct.jsonl → Fine-tune → GGUF

BTC Physics CSV
    ↓
scripts/convert_physics_to_sympy.py (deterministic formula extraction; no closed-source LLM)
    ↓
data/distilled/physics_kb.formulas.jsonl

PhysicsFormulae GitHub
    ↓
scripts/convert_physics_to_sympy.py (curate + format)
    ↓
data/distilled/physics_kb.from_pf.jsonl

data/distilled/*.jsonl
    ↓
scripts/rag/build_physics_index.py (embed BGE-M3 → LlamaIndex)
    ↓
storage/vector_db/ → physics_rag_node (runtime)
```

---

## 8. GHI CHÚ

- **Fine-tune datasets**: Đã filter Q19 (drop rows có `id` bắt đầu bằng `QA`)
- **Electro**: Dataset nội bộ (textbook điện từ), không phải từ BTC hay GitHub
- **PhysicsFormulae**: Facts (công thức) không bản quyền, LaTeX đã transform sang plain math
- **Closed-source LLM use**: Không dùng GPT/Claude/Gemini hoặc API thương mại cho training, preprocessing, RAG hay inference
- **FOLIO**: Dataset logic từ Yale NLP (https://huggingface.co/datasets/yale-nlp/FOLIO)
