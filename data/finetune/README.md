# EXACT 2026 — Fine-tune datasets

Thư mục này chứa 2 bộ dữ liệu ChatML đã sẵn sàng để upload lên Google Colab/Drive
và chạy fine-tune với Unsloth (hoặc bất kỳ trainer nào nhận `messages`-format).

Hai bộ phục vụ **2 model khác nhau** trong pipeline 2-specialist của EXACT 2026:

| Bộ dữ liệu      | Model fine-tune                  | Vai trò trong runtime                                        |
| --------------- | -------------------------------- | ------------------------------------------------------------ |
| `coder.*`       | **Qwen2.5-Coder-7B-Instruct**    | Sinh **code Z3/SymPy** từ premises/question                  |
| `instruct.*`    | **Qwen2.5-7B-Instruct**          | Sinh **JSON `ExactResponse`** từ code + output của solver    |

> [!NOTE]
> Cả 2 model đều ≤ 8B (theo Q1 BTC), share Qwen tokenizer family, dùng cùng
> ChatML template nên có thể tái sử dụng prompt và pipeline đánh giá.
> Tại runtime chỉ **1 model resident** ở một thời điểm (Q3 BTC) —
> `llama-server` swap process Coder ↔ Instruct giữa các giai đoạn.

---

## Files

### 1. `coder.jsonl` (~3.3 MB · 1,391 records)

- **Mục đích**: train model Coder sinh code thuần (Z3 cho Type 1, SymPy cho Type 2).
- **Format mỗi dòng**:
  ```json
  {
    "messages": [
      {"role": "system",    "content": "You are an expert solver for the EXACT 2026 ..."},
      {"role": "user",      "content": "[LOGIC PROBLEM] ... Premises: ... Question: ..."},
      {"role": "assistant", "content": "```python\n<Z3 hoặc SymPy code>\n```"}
    ],
    "meta": {"source": "btc_physics|folio|electro", "type": "logic|physics", "uid": "..."}
  }
  ```
- **Nguồn dữ liệu**:
  | Source        | Records | Mô tả                                                                |
  | ------------- | ------: | -------------------------------------------------------------------- |
  | `btc_physics` |   1,022 | Dataset BTC chính thức - Physics CSV → SymPy template               |
  | `folio`       |     177 | Dataset bên ngoài (yale-nlp/FOLIO) - premises-FOL → Z3 entailment |
  | `electro`     |     192 | Dataset nội bộ - Textbook điện từ → SymPy code (đã verify)         |

  *Mọi record đều đã pass `exec()` filter — code thực sự chạy được.*
- **Model đích**: `unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit`

### 2. `coder.eval.jsonl` (~390 KB · 155 records)

- 10% holdout của `coder.jsonl`, dùng làm validation split (eval loss / sample sinh).
- Cùng format & meta như trên.

### 3. `instruct.jsonl` (~14 MB · 2,518 records)

- **Mục đích**: train model Instruct nhận (problem + code + code_output) và xuất
  **một JSON object** đúng schema `ExactResponse` của API EXACT.
- **Format mỗi dòng**:
  ```json
  {
    "messages": [
      {"role": "system",    "content": "You are the explanation engine ..."},
      {"role": "user",      "content": "[LOGIC|PHYSICS PROBLEM] ... Solver code: ```...``` Solver stdout/error: ```...```"},
      {"role": "assistant", "content": "{\"answer\":\"...\",\"explanation\":\"...\",\"fol\":\"...\"|null,\"cot\":[\"...\",\"...\"]|null,\"premises\":[\"Premise 1: ...\"],\"confidence\":0.92}"}
    ],
    "meta": {"source": "...", "type": "logic|physics", "branch": "success|error", "uid": "..."}
  }
  ```
- **Hai nhánh prompt** (mirror runtime branching trong `src/agent/nodes/*_explanation.py`):
  - `branch = success` → solver chạy ok, code_output là stdout thật → confidence cao (~0.9)
  - `branch = error`   → solver fail, code_output là error string → fallback reasoning → confidence ~0.6
- **`code_output` đến từ đâu?** Xem section [Cách sinh `code_output` cho instruct dataset](#cách-sinh-code_output-cho-instruct-dataset) phía dưới.
- **Nguồn dữ liệu**:
  | Source        | Records | Mô tả                                                                 |
  | ------------- | ------: | --------------------------------------------------------------------- |
  | `btc_physics` |   1,213 | Dataset BTC chính thức - Physics + SymPy code + cot                  |
  | `folio`       |   1,082 | Dataset bên ngoài (yale-nlp/FOLIO) - premises + Z3 code + label      |
  | `electro`     |     223 | Dataset nội bộ - Textbook điện từ + SymPy code                       |

  *Tỷ lệ branch thực tế ~58% error / 42% success* (vì FOLIO Z3 có ~44% record Z3-engine
  thật sự fail nên cũng route vào nhánh error). Có thể giảm bằng `--error-ratio 0.0`
  rồi điều chỉnh.
- **Model đích**: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`

### 4. `instruct.eval.jsonl` (~1.5 MB · 280 records)

- 10% holdout của `instruct.jsonl`.
- Dùng để theo dõi eval loss và verify model giữ format JSON đúng.

### 5. `coder.STATS.md` & `instruct.STATS.md`

- Báo cáo tự sinh: phân phối theo source / type / branch + trung bình / P95 độ dài.
- Cập nhật mỗi lần chạy lại pipeline.

---

## Cách sinh `code_output` cho instruct dataset

`code_output` được tạo **offline lúc build dataset**, không lấy từ runtime. Có 2 nguồn:

### Nguồn 1 — Lỗi thật khi `exec()` code

Trong `scripts/data_prep/_common.py`, hàm `verify_python()` chạy code trong namespace
cô lập, capture stdout/stderr, và trả về `ExecResult`:

```python
def verify_python(code: str) -> ExecResult:
    out_buf, err_buf = io.StringIO(), io.StringIO()
    ns = {"__builtins__": __builtins__}
    try:
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            exec(code, ns)
    except Exception as e:
        return ExecResult(ok=False, stdout=..., 
                          error=f"{type(e).__name__}: {e}")  # ← error string thật
    return ExecResult(ok=True, stdout=out_buf.getvalue(), error="")
```

Khi code Z3/SymPy thật sự fail (ví dụ FOLIO có conversion sai cú pháp
`s.add(ForAll(x, P(x)))))`), exception sẽ được capture thành chuỗi như
`"SyntaxError: unmatched ')'"` hoặc `"NameError: name 'foo' is not defined"`.

### Nguồn 2 — Synthetic (ép vào nhánh error)

Trong `scripts/data_prep/prepare_instruct_dataset.py`, mỗi builder có pattern:

```python
force_error = rng.random() < error_ratio   # default 30%
res = verify_python(code)
if force_error:
    code_error = True
    code_output = res.error or "RuntimeError: solver inconclusive"
else:
    code_error = not res.ok
    code_output = res.stdout if res.ok else (res.error or "")
```

**Ma trận quyết định:**

| `force_error` | Code chạy ok? | `code_error` | `code_output`                                           |
| :-----------: | :-----------: | :----------: | ------------------------------------------------------- |
|     True      |      Yes      |     True     | fallback string `"RuntimeError: solver inconclusive"`   |
|     True      |      No       |     True     | error string thật từ `verify_python`                    |
|     False     |      Yes      |     False    | stdout thật                                             |
|     False     |      No       |     True     | error string thật                                       |

→ Đó là lý do tỷ lệ branch thực tế ~58% chứ không phải đúng 30%: lỗi tự nhiên
(~44% với FOLIO Z3 sau filter) cộng dồn với synthetic 30% → tổng ~58%.

### Điểm chèn vào prompt

Hàm `_build_user_msg()` trong `prepare_instruct_dataset.py` chèn `code_output`
vào user message theo pattern:

```
Solver code:
```python
<code đã sinh>
```

Solver stdout (SUCCESS branch):     ← hoặc "Solver error (ERROR branch)"
```
<code_output>                       ← chèn ở đây
```

Produce the ExactResponse JSON object now.
```

Target assistant cũng được sinh tương ứng (`_logic_target` / `_physics_target`):
nếu `code_error=True` thì explanation acknowledge "the solver failed" và fallback
reasoning từ premises/cot, confidence hạ xuống 0.6.

### Tổng kết flow

```
1. Sinh code (folio_to_z3 / generate_sympy_code / electro pre-existing)
        ↓
2. verify_python(code) → res.ok / res.error / res.stdout
        ↓
3. force_error = rng.random() < error_ratio
        ↓
4. Quyết định code_error + code_output (theo bảng trên)
        ↓
5. _build_user_msg(...) chèn code_output vào user prompt
        ↓
6. _logic_target / _physics_target sinh JSON ExactResponse target
        ↓
7. chatml(system, user, assistant) → ghi vào instruct.jsonl
```

---

## Cách regenerate

```powershell
# Coder dataset
.\venv\Scripts\python.exe -m scripts.data_prep.prepare_coder_dataset

# Instruct dataset
.\venv\Scripts\python.exe -m scripts.data_prep.prepare_instruct_dataset

# Tuỳ chọn
#   --no-verify          bỏ exec() filter (nhanh hơn ~5x)
#   --error-ratio 0.20   giảm tỷ lệ nhánh error (default 0.30)
#   --val-ratio 0.05     giảm holdout xuống 5%
#   --no-electro         bỏ nguồn electro
```

---

## Workflow đề xuất trên Colab

1. Zip `data/finetune/` → upload Google Drive.
2. Notebook 1 — Coder
   - Load `Qwen2.5-Coder-7B-Instruct-bnb-4bit` qua Unsloth.
   - Train 2-3 epoch với `coder.jsonl`, eval bằng `coder.eval.jsonl`.
   - Export GGUF Q4_K_M → drop vào `models/qwen-coder-7b.gguf`.
3. Notebook 2 — Instruct
   - Load `Qwen2.5-7B-Instruct-bnb-4bit` qua Unsloth.
   - Train 2-3 epoch với `instruct.jsonl`, eval bằng `instruct.eval.jsonl`.
   - Export GGUF Q4_K_M → drop vào `models/qwen-instruct-7b.gguf`.
4. Cập nhật `config/setting.yaml` để `LlamaServerSupervisor` swap đúng 2 GGUF mới.

---

## Code fine-tune

Code fine-tune nằm tại `fine_tune/`:
- `fine_tune/qwen2.5-coder-7b/fine_tune.ipynb` — Notebook fine-tune Coder
- `fine_tune/qwen2.5-7b-instruct/fine_tune.ipynb` — Notebook fine-tune Instruct

Xem `fine_tune/README.md` để biết chi tiết workflow fine-tune trên Google Colab.

---

## Schema `ExactResponse` (đối chiếu với `src/api/schemas.py` + EXACT_Slides.pdf p.33)

```json
{
  "answer": "string",
  "explanation": "string",
  "fol":   "string"  | null,
  "cot":   ["string", ...] | null,
  "premises": ["Premise 1: ...", "Premise 2: ..."],
  "confidence": 0.0-1.0
}
```

- Logic problems: `fol` populated (single FOL formula string của conclusion), `cot` = null.
- Physics problems: `cot` populated (list các step strings tách từ CoT), `fol` = null.
- Hệ thống chấm sẽ parse JSON này để tính P1 (correctness) + P2 (explanation) + P3 (reasoning depth).
