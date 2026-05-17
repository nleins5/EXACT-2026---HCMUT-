# EXACT 2026 — Scripts

Bộ công cụ xử lý dữ liệu cho cuộc thi **EXACT 2026**, phục vụ fine-tune model `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`.

## Tổng quan

```
scripts/
├── build_final_dataset.py        # Pipeline chính (chạy 1 lệnh = xong)
├── convert_physics_to_sympy.py   # Chuyển bài Vật lý → mã SymPy
├── convert_logic_to_z3.py        # Chuyển bài Logic (FOLIO) → mã Z3
└── README.md                     # File này
```

## Quick Start

```bash
# Chạy full pipeline — tự động tạo dataset hoàn chỉnh
python scripts/build_final_dataset.py
```

Một lệnh duy nhất sẽ:

1. **Generate SymPy** — chuyển 242 bài Vật lý → code SymPy, verify executable
2. **Generate Z3** — tải FOLIO từ HuggingFace, chuyển FOL → code Z3
3. **Load BTC** — đọc data chính thức ban tổ chức (Logic JSON + Physics CSV)
4. **Filter & Merge** — lọc code chạy được, gộp tất cả nguồn
5. **Export** — chia train/val (90/10), ghi ra `data/colab_ready/`

**Output:**

```
data/colab_ready/
├── train.jsonl    # ~2489 samples
└── val.jsonl      # ~276 samples
```

---

## Chi tiết từng Script

### 1. `build_final_dataset.py` — Pipeline chính

**Mục đích:** Chạy toàn bộ pipeline từ đầu đến cuối, tạo ra dataset sẵn sàng fine-tune.

**Cách dùng:**

```bash
# Full pipeline (BTC + SymPy + Z3)
python scripts/build_final_dataset.py

# Chỉ dùng data BTC (không augment)
python scripts/build_final_dataset.py --btc-only

# Bỏ qua SymPy (chỉ BTC + Z3)
python scripts/build_final_dataset.py --no-sympy

# Bỏ qua Z3 (chỉ BTC + SymPy)
python scripts/build_final_dataset.py --no-z3
```

**Pipeline flow:**

```
┌─────────────────────────────────────────────────────────────┐
│                    build_final_dataset.py                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 1: SymPy Generation                                   │
│  ┌─────────────────────┐     ┌──────────────────────────┐   │
│  │ electro_dataset.jsonl│────▶│ convert_physics_to_sympy │   │
│  │ (242 bài vật lý)    │     │ .generate_sympy_code()   │   │
│  └─────────────────────┘     └──────────┬───────────────┘   │
│                                         │ 217 executable    │
│  Step 2: Z3 Generation                  │                   │
│  ┌─────────────────────┐     ┌──────────▼───────────────┐   │
│  │ FOLIO (HuggingFace) │────▶│ convert_logic_to_z3      │   │
│  │ (yale-nlp/FOLIO)    │     │ .fol_to_z3_code()        │   │
│  └─────────────────────┘     └──────────┬───────────────┘   │
│                                         │ 1204 records      │
│  Step 3: BTC Official Data              │                   │
│  ┌─────────────────────┐                │                   │
│  │ Logic JSON (808 q)  │───┐            │                   │
│  └─────────────────────┘   │            │                   │
│  ┌─────────────────────┐   │            │                   │
│  │ Physics CSV (1352 q)│───┤            │                   │
│  └─────────────────────┘   │            │                   │
│                            │            │                   │
│  Step 4: Filter & Merge    ▼            ▼                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  exec() filter → merge → shuffle (seed=3407)       │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         │                                   │
│  Step 5: Export         ▼                                   │
│  ┌──────────────────────────────┐                           │
│  │ data/colab_ready/train.jsonl │  ~2489 samples            │
│  │ data/colab_ready/val.jsonl   │  ~276 samples             │
│  └──────────────────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

**Output format** (mỗi dòng JSONL):

```json
{
  "conversations": [
    {"role": "system", "content": "You are an expert educational AI..."},
    {"role": "user", "content": "[LOGIC PROBLEM]\nPremises:\n1. ..."},
    {"role": "assistant", "content": "<think>\n...\n</think>\n<answer>\n...\n</answer>"}
  ]
}
```

---

### 2. `convert_physics_to_sympy.py` — Chuyển Vật lý → SymPy

**Mục đích:** Chuyển đổi bài toán Vật lý (điện từ) từ LaTeX sang mã SymPy executable. Được `build_final_dataset.py` gọi tự động, nhưng cũng có thể chạy standalone để debug/preview.

**Cách dùng (standalone):**

```bash
# Tạo mã SymPy cho tất cả 242 bài
python scripts/convert_physics_to_sympy.py --generate --count 242

# Kiểm tra bao nhiêu bài chạy được
python scripts/convert_physics_to_sympy.py --verify

# Phân tích lỗi (phân loại pattern)
python scripts/convert_physics_to_sympy.py --analyze

# Xem trước 5 bài đầu
python scripts/convert_physics_to_sympy.py --preview 5
```

**Các mode:**

| Mode | Mô tả |
|------|--------|
| `--generate` | Đọc `electro_dataset.jsonl`, chuyển LaTeX → SymPy, ghi ra `electro_sympy_dataset.jsonl` |
| `--verify` | Chạy `exec()` từng record, báo cáo tỷ lệ pass/fail |
| `--analyze` | Phân loại lỗi (SyntaxError, TypeError, ...) theo pattern |
| `--preview N` | In ra N bài đầu tiên để kiểm tra bằng mắt |

**Input/Output:**

```
Input:  data/collected/electro_dataset.jsonl       (242 bài từ textbook điện từ)
Output: data/collected/electro_sympy_dataset.jsonl  (242 bài + sympy_verify_code)
```

**Conversion engine:**
- Heuristic LaTeX → SymPy (regex-based)
- Xử lý: `\frac{}{}`, `\sqrt{}`, `^{}`, trig functions, Greek symbols, implicit multiplication
- Tự skip biểu thức quá phức tạp (aligned environments, matrices, ...)

---

### 3. `convert_logic_to_z3.py` — Chuyển Logic (FOLIO) → Z3

**Mục đích:** Tải dataset FOLIO (First-Order Logic) từ HuggingFace, chuyển mỗi bài thành script Z3 theorem prover. Được `build_final_dataset.py` gọi tự động, nhưng cũng có thể chạy standalone.

**Cách dùng (standalone):**

```bash
# Tạo Z3 dataset (tải FOLIO + convert)
python scripts/convert_logic_to_z3.py

# Chỉ định output directory khác
python scripts/convert_logic_to_z3.py --output-dir ./my_output
```

**Input/Output:**

```
Input:  yale-nlp/FOLIO (HuggingFace, tự download + cache)
Output: data/sft_dataset/train.jsonl  (Alpaca format, ~1083 records)
        data/sft_dataset/val.jsonl    (Alpaca format, ~121 records)
```

**Conversion engine:**
- FOL → Z3 Python: `∀x` → `ForAll(x, ...)`, `∃x` → `Exists(x, ...)`
- Operators: `¬` → `Not()`, `∧` → `And()`, `∨` → `Or()`, `→` → `Implies()`
- Auto-detect predicates (uppercase) và constants (lowercase)
- Check entailment: nếu `Not(conclusion)` là `unsat` → True, etc.

---

## Nguồn dữ liệu

| Nguồn | Loại | Số lượng | Mô tả |
|-------|------|----------|-------|
| **BTC Logic** | Logic | 808 | Data chính thức ban tổ chức (JSON) |
| **BTC Physics** | Physics | 1,352 | Data chính thức ban tổ chức (CSV) |
| **FOLIO → Z3** | Logic (augmented) | 388* | Yale NLP FOLIO, chuyển sang Z3 code |
| **Electro → SymPy** | Physics (augmented) | 217* | Textbook điện từ, chuyển sang SymPy code |
| **Tổng** | | **~2,765** | |

> *Số lượng sau khi filter executable. Raw: 1204 Z3, 242 SymPy.

## Thông số kỹ thuật

- **System prompt:** Hướng dẫn model trả lời theo format `<think>...</think>` + `<answer>...</answer>`
- **Random seed:** 3407 (reproducible)
- **Train/Val split:** 90/10
- **Executable filter:** Mọi augmented code đều phải `exec()` thành công mới được giữ

## Yêu cầu

```
pip install datasets z3-solver sympy
```

Hoặc dùng venv đã có sẵn:

```bash
# Windows
& "d:\Exact 2026\venv\Scripts\python.exe" scripts/build_final_dataset.py
```

## Cấu trúc dữ liệu

```
data/
├── EXACT2026_dataset_2026-05-15/    # Data BTC (không chỉnh sửa)
│   ├── Logic_Based_.../
│   │   └── Logic_Based_Educational_Queries.json
│   └── Physics_Problems_.../
│       └── Physics_Problems_Text_Only.csv
├── collected/
│   ├── electro_dataset.jsonl         # Raw textbook data (input cho SymPy)
│   └── electro_sympy_dataset.jsonl   # Intermediate SymPy output
├── sft_dataset/                      # Intermediate augmented data
│   ├── train.jsonl                   # (tự tạo bởi pipeline)
│   └── val.jsonl
└── colab_ready/                      # ← OUTPUT CUỐI CÙNG
    ├── train.jsonl                   # Upload lên Google Drive
    └── val.jsonl                     # → Fine-tune trên Colab
```
