# data/

Thư mục chứa toàn bộ dữ liệu cho project EXACT-2026.

## Cấu trúc

```
data/
├── EXACT2026_dataset_2026-05-15/   # Dataset gốc BTC (không chỉnh sửa)
├── collected/                       # Dữ liệu thu thập từ bên ngoài
├── distilled/                       # Physics KB cho RAG (output từ scripts/distill/)
├── external/                        # Cache external resources
├── finetune/                        # Datasets ChatML cho fine-tune
├── EXACT_Slides.pdf                 # Slide BTC chính thức
├── QA.pdf                           # Q&A BTC (Q1–Q21)
└── README.md                        # File này
```

---

## EXACT2026_dataset_2026-05-15/

Dataset chính thức từ Ban Tổ Chức (BTC). **Giữ nguyên gốc, không chỉnh sửa.**

| File | Mô tả |
|------|--------|
| `Physics_Problems_Text_Only/Physics_Problems_Text_Only.csv` | 1,352 bài vật lý Type 2 (question, cot, answer, unit) |
| `Logic_Based_Educational_Queries_Text_Only/Logic_Based_Educational_Queries.json` | 411 records / 808 câu hỏi Type 1 (premises NL + FOL) |
| `CHANGELOG_TYPE1.md` / `CHANGELOG_TYPE2.md` | Lịch sử fix bug từ BTC |

**Nguồn**: BTC EXACT 2026 (phát hành 2026-05-15).

---

## collected/

Dữ liệu thu thập từ **bên ngoài** (không phải BTC), phục vụ training và RAG.

| File | Records | Nguồn gốc | Tạo bởi script |
|------|---------|------------|-----------------|
| `electro_dataset.jsonl` | 242 | Textbook điện từ (thu thập thủ công) | — (raw data) |
| `electro_sympy_dataset.jsonl` | 242 | Sinh từ `electro_dataset.jsonl` | `scripts/convert_physics_to_sympy.py --generate` |
| `train.jsonl` | 1,083 | [yale-nlp/FOLIO](https://huggingface.co/datasets/yale-nlp/FOLIO) (HuggingFace) | `scripts/convert_logic_to_z3.py --output-dir data/collected` |
| `val.jsonl` | 121 | [yale-nlp/FOLIO](https://huggingface.co/datasets/yale-nlp/FOLIO) (HuggingFace) | (cùng script trên, validation split 10%) |

**Chi tiết:**
- `electro_dataset.jsonl`: Bài toán điện từ từ sách giáo khoa, format JSON (`id`, `questions`, `solutions`, `final_answers`, `graphs`).
- `electro_sympy_dataset.jsonl`: Phiên bản đã thêm SymPy verification code (sinh bằng heuristic converter).
- `train.jsonl` / `val.jsonl`: FOLIO FOL premises → executable Z3 Python code. Format Alpaca (`instruction`, `input`, `output`, `source`, `id`).

---

## distilled/

Knowledge base cho **physics RAG node**. Output được tạo tự động bằng scripts.

| File | Records | Nguồn gốc | Tạo bởi script |
|------|---------|------------|-----------------|
| `physics_kb.formulas.jsonl` | 363 | BTC Physics CSV (1,352 bài) | `scripts/distill/extract_formulas.py` (Gemini 2.5 Flash-Lite API) |
| `physics_kb.from_pf.jsonl` | 29 | [PhysicsFormulae](https://github.com/BenjaminTMilnes/PhysicsFormulae) | `scripts/distill/fetch_physics_formulae.py` |

**Pipeline tái tạo:**
```powershell
# Extract formulas từ BTC CSV qua Gemini API
python -m scripts.distill.extract_formulas run --batch-size 50

# Build RAG index (Qdrant)
python -m scripts.rag.build_physics_index --input data/distilled/physics_kb.formulas.jsonl --rebuild
```

Xem thêm: `distilled/README.md`

---

## external/

Cache external resources (download 1 lần, dùng lại).

| File | Nguồn gốc |
|------|------------|
| `PhysicsFormulae_Compiled.json` | [BenjaminTMilnes/PhysicsFormulae](https://github.com/BenjaminTMilnes/PhysicsFormulae) — file `Compiled.json` (~655KB) |

**Dùng bởi**: `scripts/distill/fetch_physics_formulae.py`

---

## finetune/

Datasets đã preprocess (ChatML format) cho fine-tune model.

| File | Records | Mô tả | Tạo bởi script |
|------|---------|--------|-----------------|
| `coder.jsonl` | ~1,391 | Train Coder (Z3 + SymPy) | `scripts/data_prep/prepare_coder_dataset.py` |
| `coder.eval.jsonl` | ~155 | Validation split 10% | (cùng script trên) |
| `instruct.jsonl` | ~2,518 | Train Instruct (JSON response) | `scripts/data_prep/prepare_instruct_dataset.py` |
| `instruct.eval.jsonl` | ~280 | Validation split 10% | (cùng script trên) |

**Nguồn dữ liệu**: BTC Physics + BTC Logic + Electro collected + FOLIO.

---

## Quy tắc commit

- ✅ Commit: `README.md`, `physics_kb.from_pf.jsonl`, changelogs
- ❌ Ignore: dataset lớn (BTC, finetune), output regen được, logs, models
