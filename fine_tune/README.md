# fine_tune/

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

## Mục đích

Thư mục này chứa notebook Google Colab để fine-tune 2 model:

| Model | Dataset | Mục đích |
|-------|---------|----------|
| `qwen2.5-coder-7b` | `data/finetune/coder.jsonl` | Sinh code Z3/SymPy |
| `qwen2.5-7b-instruct` | `data/finetune/instruct.jsonl` | Sinh JSON ExactResponse |

## Cách dùng

1. Upload notebook vào Google Colab
2. Upload dataset từ `data/finetune/` lên Drive
3. Chạy notebook để fine-tune
4. Export GGUF → drop vào `models/`

## Dataset

- **Coder**: `data/finetune/coder.jsonl` (~1,391 records)
- **Instruct**: `data/finetune/instruct.jsonl` (~2,518 records)

Xem `data/finetune/README.md` để biết chi tiết dataset.

## Sau fine-tune

Cập nhật `config/setting.yaml` để swap GGUF đã fine-tune:

```yaml
llm:
  coder:
    model_path: models/qwen2.5-coder-7b-instruct.Q4_K_M.gguf
  instruct:
    model_path: models/qwen2.5-7b-instruct.Q4_K_M.gguf
```
