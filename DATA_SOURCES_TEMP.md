# EXACT-2026 Data Sources - Chi tiết đầy đủ

Tài liệu này mô tả chi tiết từng dataset được sử dụng trong project EXACT-2026, bao gồm:
- Mục đích sử dụng (Fine-tune vs RAG)
- Nguồn gốc (BTC, FOLIO, Physics, Electro, PhysicsFormulae)
- File code tạo ra dataset
- Vị trí lưu trữ

---

## 1. DATASET CHO FINE-TUNE

### 1.1. Coder Dataset (Qwen2.5-Coder-7B-Instruct)

**Mục đích**: Train model sinh code Z3 (logic) hoặc SymPy (vật lý).

**Files**:
- `data/finetune/coder.jsonl` (~1,391 records)
- `data/finetune/coder.eval.jsonl` (~155 records - 10% holdout)

**Nguồn dữ liệu**:

| Source | Records | Mô tả | File code tạo |
|--------|---------|-------|---------------|
| **BTC Physics** | 1,022 | Dataset BTC chính thức - Physics CSV → SymPy template | `scripts/data_prep/prepare_coder_dataset.py` |
| **FOLIO** | 177 | Dataset FOLIO (yale-nlp/FOLIO) - premises-FOL → Z3 entailment | `scripts/data_prep/prepare_coder_dataset.py` |
| **Electro** | 192 | Textbook điện từ (nội bộ) → SymPy code | `scripts/data_prep/prepare_coder_dataset.py` |

**Workflow**:
```powershell
.\venv\Scripts\python.exe -m scripts.data_prep.prepare_coder_dataset
```

**Output**:
- `data/finetune/coder.jsonl` - Training split
- `data/finetune/coder.eval.jsonl` - Validation split
- `data/finetune/coder.STATS.md` - Statistics

---

### 1.2. Instruct Dataset (Qwen2.5-7B-Instruct)

**Mục đích**: Train model sinh JSON `ExactResponse` từ (problem + code + code_output).

**Files**:
- `data/finetune/instruct.jsonl` (~2,518 records)
- `data/finetune/instruct.eval.jsonl` (~280 records - 10% holdout)

**Nguồn dữ liệu**:

| Source | Records | Mô tả | File code tạo |
|--------|---------|-------|---------------|
| **BTC Physics** | 1,213 | Dataset BTC chính thức - Physics + SymPy code + cot | `scripts/data_prep/prepare_instruct_dataset.py` |
| **FOLIO** | 1,082 | Dataset FOLIO - premises + Z3 code + label | `scripts/data_prep/prepare_instruct_dataset.py` |
| **Electro** | 223 | Textbook điện từ - SymPy code | `scripts/data_prep/prepare_instruct_dataset.py` |

**Workflow**:
```powershell
.\venv\Scripts\python.exe -m scripts.data_prep.prepare_instruct_dataset
```

**Output**:
- `data/finetune/instruct.jsonl` - Training split
- `data/finetune/instruct.eval.jsonl` - Validation split
- `data/finetune/instruct.STATS.md` - Statistics

---

## 2. DATASET CHO RAG (Physics Knowledge Base)

### 2.1. Physics KB cho RAG

**Mục đích**: Cung cấp công thức vật lý và ví dụ mẫu cho `physics_rag_node`.

**Files**:

| File | Records | Trạng thái | Mô tả |
|------|---------|------------|-------|
| `data/distilled/physics_kb.from_pf.jsonl` | 28 | ✅ Commit | Công thức từ PhysicsFormulae (verified=true) |
| `data/distilled/physics_kb.raw.jsonl` | ~1,594 | ❌ Ignore | Output thô teacher LLM |
| `data/distilled/physics_kb.verified.jsonl` | ~1,594 | ❌ Ignore | Sau khi exec SymPy |

**Nguồn dữ liệu**:

| Source | Records | Mô tả | File code tạo |
|--------|---------|-------|---------------|
| **PhysicsFormulae GitHub** | 22 formulas + 6 constants | External resource - PhysicsFormulae | `scripts/distill/fetch_physics_formulae.py` |
| **BTC Physics** | 1,352 | Dataset BTC chính thức - Physics CSV | `scripts/distill/distill_physics.py` |
| **Electro** | 242 | Textbook điện từ (nội bộ) | `scripts/distill/distill_physics.py` |

**Workflow**:
```powershell
# 1. Pull công thức từ PhysicsFormulae
.\venv\Scripts\python.exe -m scripts.distill.fetch_physics_formulae --include-constants

# 2. Distill từ BTC + electro (Gemini Flash Lite)
$env:GOOGLE_API_KEY = "<your-key>"
.\venv\Scripts\python.exe -m scripts.distill.distill_physics --source all

# 3. Verify SymPy code
.\venv\Scripts\python.exe -m scripts.distill.verify_kb

# 4. Build Qdrant index
.\venv\Scripts\python.exe -m scripts.rag.build_physics_index --rebuild
```

**Output**:
- `physics_kb.from_pf.jsonl` - 28 records (22 formulas + 6 constants), verified=true
- `physics_kb.raw.jsonl` - Output thô teacher LLM
- `physics_kb.verified.jsonl` - Sau khi exec SymPy, mark verified=true/false

**Collections Qdrant**:
- `physics_examples` - per-record (dùng khi query giống 1 bài cụ thể)
- `physics_formulas` - per-topic (dùng khi query mới → fallback formula sheet)

---

## 3. DATASET GỐC (BTC)

### 3.1. Logic-Based Educational Queries (Type 1)

**File**: `data/EXACT2026_dataset_2026-05-15/Logic_Based_Educational_Queries_Text_Only/Logic_Based_Educational_Queries.json`

**Records**: 411 records / 808 câu hỏi

**Fields**:
- `premises-NL`: Các giả thiết bằng tiếng Anh
- `premises-FOL`: First-Order Logic
- `question`: Câu hỏi
- `answer`: Đáp án (Yes/No/Unknown hoặc A/B/C/D)
- `explanation`: Giải thích

**Mục đích**: Nguồn gốc cho fine-tune Coder (logic) và Instruct (logic)

---

### 3.2. Physics Problems (Type 2)

**File**: `data/EXACT2026_dataset_2026-05-15/Physics_Problems_Text_Only/Physics_Problems_Text_Only.csv`

**Records**: 1,352 bài (đã filter `QA-`)

**Fields**:
- `id`: ID bài toán
- `question`: Câu hỏi
- `cot`: Chain-of-Thought reasoning
- `answer`: Đáp án số
- `unit`: Đơn vị SI

**Mục đích**: Nguồn gốc cho fine-tune Coder (physics) và Instruct (physics)

---

## 4. DATASET NỘI BỘ (Electro)

### 4.1. Electro Dataset

**File**: `data/collected/electro_dataset.jsonl`

**Records**: 242 bài điện từ

**Fields**:
- `id`: ID bài toán
- `questions`: Câu hỏi
- `solutions`: Lời giải chi tiết
- `final_answers`: Đáp án cuối cùng
- `graphs`: Đồ thị (nếu có)

**Nguồn**: Textbook điện từ (tự thu thập)

**Mục đích**: Nguồn bổ sung cho fine-tune Coder và Instruct (physics)

---

### 4.2. Electro Sympy Dataset

**File**: `data/collected/electro_sympy_dataset.jsonl`

**Records**: 242 bài + SymPy code đã verify

**Fields**:
- `id`: ID bài toán
- `questions`: Câu hỏi
- `solution`: Lời giải
- `final_answers`: Đáp án
- `sympy_code`: Code SymPy đã verify

**Mục đích**: Code SymPy mẫu cho fine-tune Coder (physics)

---

## 5. EXTERNAL RESOURCES

### 5.1. PhysicsFormulae

**File**: `data/external/PhysicsFormulae_Compiled.json`

**Records**: ~655KB cache từ [BenjaminTMilnes/PhysicsFormulae](https://github.com/BenjaminTMilnes/PhysicsFormulae)

**Mục đích**: Nguồn công thức vật lý chuẩn cho RAG

**Cách dùng**: `scripts/distill/fetch_physics_formulae.py` crawl và convert LaTeX → plain math

---

## 6. SUMMARY TABLE

| Dataset | Mục đích | Nguồn | File code tạo | Records |
|---------|----------|------|---------------|---------|
| **Fine-tune Coder** | Train Coder sinh code | BTC + FOLIO + Electro | `scripts/data_prep/prepare_coder_dataset.py` | ~1,391 |
| **Fine-tune Instruct** | Train Instruct sinh JSON | BTC + FOLIO + Electro | `scripts/data_prep/prepare_instruct_dataset.py` | ~2,518 |
| **Physics KB (from PF)** | RAG formulas | PhysicsFormulae GitHub | `scripts/distill/fetch_physics_formulae.py` | 28 |
| **Physics KB (raw)** | RAG examples | BTC + Electro | `scripts/distill/distill_physics.py` | ~1,594 |
| **Physics KB (verified)** | RAG examples | BTC + Electro | `scripts/distill/verify_kb.py` | ~1,594 |

---

## 7. FLOW DIAGRAM

```
BTC Data (Physics CSV + Logic JSON)
    ↓
scripts/data_prep/prepare_coder_dataset.py
    ↓
data/finetune/coder.jsonl → Fine-tune Coder → GGUF

BTC Data (Physics CSV + Logic JSON)
    ↓
scripts/data_prep/prepare_instruct_dataset.py
    ↓
data/finetune/instruct.jsonl → Fine-tune Instruct → GGUF

PhysicsFormulae GitHub
    ↓
scripts/distill/fetch_physics_formulae.py
    ↓
data/distilled/physics_kb.from_pf.jsonl → Build Qdrant index

BTC + Electro
    ↓
scripts/distill/distill_physics.py (Gemini API)
    ↓
data/distilled/physics_kb.raw.jsonl
    ↓
scripts/distill/verify_kb.py
    ↓
data/distilled/physics_kb.verified.jsonl → Build Qdrant index

Qdrant index → physics_rag_node (runtime)
```

---

## 8. GHI CHÚ

- **Fine-tune datasets**: Đã filter Q19 (drop rows có `id` bắt đầu bằng `QA`)
- **Physics KB**: Chỉ 2 topic theo scope EXACT 2026: `electrostatics` + `electric_circuits`
- **Electro**: Dataset nội bộ (textbook điện từ), không phải từ BTC hay GitHub
- **PhysicsFormulae**: Không có LICENSE → facts (công thức) không bản quyền, LaTeX đã transform sang plain math
