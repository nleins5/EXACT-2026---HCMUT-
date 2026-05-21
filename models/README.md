# models/

Chứa GGUF model files cho llama-server.

## Cấu trúc

```
models/
├── download_models.py       # Script tải model từ HuggingFace
└── exact-2026/              # Thư mục model (từ HF repo)
    ├── Qwen2.5-7B-Instruct.Q4_K_M.gguf      # Instruct model (fine-tuned, ~4.7GB)
    ├── qwen2.5-coder-7b-instruct.Q4_K_M.gguf # Coder model (fine-tuned, ~4.7GB)
    ├── adapter_config.json
    ├── config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── chat_template.jinja
    ├── Modelfile
    └── README.md
```

## Download

```bash
cd models
python download_models.py
```

Script sẽ tải 2 file GGUF (~4.7GB mỗi file) từ `HoangKhangHCMUS/exact-2026`.
Nếu file đã tồn tại sẽ tự động skip.

## Model info

| File | Role | Base model | Quantization |
|------|------|-----------|--------------|
| `Qwen2.5-7B-Instruct.Q4_K_M.gguf` | instruct | Qwen2.5-7B-Instruct | Q4_K_M |
| `qwen2.5-coder-7b-instruct.Q4_K_M.gguf` | coder | Qwen2.5-Coder-7B-Instruct | Q4_K_M |

## Cấu hình

Đường dẫn model được set trong `config/setting.yaml`:
```yaml
llm:
  coder:
    model_path: models/exact-2026/qwen2.5-coder-7b-instruct.Q4_K_M.gguf
  instruct:
    model_path: models/exact-2026/Qwen2.5-7B-Instruct.Q4_K_M.gguf
```
