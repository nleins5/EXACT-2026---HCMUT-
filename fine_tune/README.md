# EXACT-2026 Fine-tuning

Thư mục này chứa code fine-tune cho 2 model trong pipeline EXACT-2026.

## Cấu trúc

```
fine_tune/
├── README.md                    # File này
├── qwen2.5-coder-7b/           # Code fine-tune Coder (Qwen2.5-Coder-7B)
│   └── fine_tune.ipynb
└── qwen2.5-7b-instruct/        # Code fine-tune Instruct (Qwen2.5-7B-Instruct)
    └── fine_tune.ipynb
```

## Dataset cho fine-tune

### 1. Dataset Coder (Qwen2.5-Coder-7B)

**Mục đích**: Train model sinh code Z3 (logic) hoặc SymPy (vật lý).

**Dataset**: `data/finetune/coder.jsonl` (~1,391 records)

**Nguồn gốc**:
| Source        | Records | Mô tả |
| ------------- | ------: | ----- |
| **BTC Physics** | 1,022 | Dataset BTC chính thức - Physics CSV → SymPy template |
| **FOLIO** | 177 | Dataset FOLIO (yale-nlp/FOLIO) - premises-FOL → Z3 entailment |
| **Electro** | 192 | Textbook điện từ (nội bộ) → SymPy code |

**Cách sinh**:
```powershell
.\venv\Scripts\python.exe -m scripts.data_prep.prepare_coder_dataset
```

**Format**:
```json
{
  "messages": [
    {"role": "system", "content": "You are an expert solver for the EXACT 2026..."},
    {"role": "user", "content": "[LOGIC|PHYSICS PROBLEM] ..."},
    {"role": "assistant", "content": "```python\n<Z3 hoặc SymPy code>\n```"}
  ],
  "meta": {"source": "btc_physics|folio|electro", "type": "logic|physics", "uid": "..."}
}
```

### 2. Dataset Instruct (Qwen2.5-7B-Instruct)

**Mục đích**: Train model sinh JSON `ExactResponse` từ (problem + code + code_output).

**Dataset**: `data/finetune/instruct.jsonl` (~2,518 records)

**Nguồn gốc**:
| Source        | Records | Mô tả |
| ------------- | ------: | ----- |
| **BTC Physics** | 1,213 | Dataset BTC chính thức - Physics + SymPy code + cot |
| **FOLIO** | 1,082 | Dataset FOLIO - premises + Z3 code + label |
| **Electro** | 223 | Textbook điện từ - SymPy code |

**Cách sinh**:
```powershell
.\venv\Scripts\python.exe -m scripts.data_prep.prepare_instruct_dataset
```

**Format**:
```json
{
  "messages": [
    {"role": "system", "content": "You are the explanation engine..."},
    {"role": "user", "content": "[LOGIC|PHYSICS PROBLEM] ... Solver code: ```...``` ..."},
    {"role": "assistant", "content": "{\"answer\":\"...\",\"explanation\":\"...\",\"fol\":...|null,\"cot\":...|null,\"confidence\":0.92}"}
  ],
  "meta": {"source": "...", "type": "logic|physics", "branch": "success|error", "uid": "..."}
}
```

**Hai nhánh**:
- `branch = success`: solver chạy ok → confidence ~0.9
- `branch = error`: solver fail → fallback reasoning → confidence ~0.6

## Workflow fine-tune trên Google Colab

### 1. Coder (Qwen2.5-Coder-7B)

1. Upload `data/finetune/coder.jsonl` và `coder.eval.jsonl` lên Google Drive
2. Mở notebook `qwen2.5-coder-7b/fine_tune.ipynb`
3. Chạy các cell:
   - Load `unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit`
   - Train 2-3 epoch với `coder.jsonl`, eval bằng `coder.eval.jsonl`
   - Export GGUF Q4_K_M
4. Download GGUF → drop vào `models/qwen2.5-coder-7b-instruct.Q4_K_M.gguf`

### 2. Instruct (Qwen2.5-7B-Instruct)

1. Upload `data/finetune/instruct.jsonl` và `instruct.eval.jsonl` lên Google Drive
2. Mở notebook `qwen2.5-7b-instruct/fine_tune.ipynb`
3. Chạy các cell:
   - Load `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`
   - Train 2-3 epoch với `instruct.jsonl`, eval bằng `instruct.eval.jsonl`
   - Export GGUF Q4_K_M
4. Download GGUF → drop vào `models/qwen2.5-7b-instruct.Q4_K_M.gguf`

## Cấu hình sau fine-tune

Sau khi có GGUF đã fine-tune, cập nhật `config/setting.yaml`:

```yaml
llm:
  server:
    binary: bin/llama-cpp/llama-server.exe
    host: 127.0.0.1
    port: 8001
    base_url: http://127.0.0.1:8001/v1
  coder:
    model_name: qwen2.5-coder-7b-instruct
    model_path: models/qwen2.5-coder-7b-instruct.Q4_K_M.gguf  # ← GGUF đã FT
    temperature: 0.0
    max_tokens: 1024
  instruct:
    model_name: qwen2.5-7b-instruct
    model_path: models/qwen2.5-7b-instruct.Q4_K_M.gguf  # ← GGUF đã FT
    temperature: 0.0
    max_tokens: 1024
```

## Liên kết với các file khác

- **`data/finetune/README.md`**: Hướng dẫn chi tiết về datasets (coder.jsonl, instruct.jsonl)
- **`scripts/data_prep/prepare_coder_dataset.py`**: Code sinh `coder.jsonl`
- **`scripts/data_prep/prepare_instruct_dataset.py`**: Code sinh `instruct.jsonl`
- **`src/agent/llm/server_supervisor.py`**: LlamaServerSupervisor swap GGUF
- **`src/app.py`**: FastAPI lifespan warm-up model

## Lưu ý

- **Không commit GGUF** vào git (file lớn). Thêm vào `.gitignore`.
- **Chỉ commit notebook** fine-tune (`.ipynb`).
- **Dataset đã filter Q19**: drop rows có `id` bắt đầu bằng `QA`.
