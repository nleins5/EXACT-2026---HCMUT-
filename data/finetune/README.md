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
    "meta": {"source": "folio|btc_physics|electro", "type": "logic|physics", "uid": "..."}
  }
  ```
- **Nguồn dữ liệu**:
  | Source        | Records | Mô tả                                                                |
  | ------------- | ------: | -------------------------------------------------------------------- |
  | `folio`       |     177 | yale-nlp/FOLIO premises-FOL → Z3 entailment script (verified `exec`) |
  | `btc_physics` |   1,022 | BTC Physics CSV (Q19-filtered) → SymPy verification template         |
  | `electro`     |     192 | Textbook điện từ → SymPy code (đã verify trong file electro_sympy)   |

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
      {"role": "assistant", "content": "{\"answer\":\"...\",\"explanation\":\"...\",\"fol\":[...]|null,\"cot\":\"...\"|null,\"premises\":[\"Premise 1: ...\"],\"confidence\":0.92}"}
    ],
    "meta": {"source": "...", "type": "logic|physics", "branch": "success|error", "uid": "..."}
  }
  ```
- **Hai nhánh prompt** (mirror runtime branching trong `src/agent/nodes/*_explanation.py`):
  - `branch = success` → solver chạy ok, code_output là stdout thật → confidence cao (~0.9)
  - `branch = error`   → solver fail, code_output là error string → fallback reasoning → confidence ~0.6
- **Nguồn dữ liệu**:
  | Source        | Records | Mô tả                                                                 |
  | ------------- | ------: | --------------------------------------------------------------------- |
  | `folio`       |   1,082 | FOLIO premises + Z3 code + label → JSON answer Yes/No/Unknown         |
  | `btc_physics` |   1,213 | BTC Physics + SymPy code + cot → JSON answer với cot làn premises     |
  | `electro`     |     223 | Textbook điện từ + SymPy code → JSON answer dạng LaTeX                |

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

## Schema `ExactResponse` (đối chiếu với `src/api/schemas.py`)

```json
{
  "answer": "string",
  "explanation": "string",
  "fol":   ["string", ...] | null,
  "cot":   "string" | null,
  "premises": ["Premise 1: ...", "Premise 2: ..."],
  "confidence": 0.0-1.0
}
```

- Logic problems: `fol` populated, `cot` = null.
- Physics problems: `cot` populated, `fol` = null.
- Hệ thống chấm sẽ parse JSON này để tính P1 (correctness) + P2 (explanation) + P3 (reasoning depth).
