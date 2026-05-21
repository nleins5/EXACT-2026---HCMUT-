# scripts/data_prep/

Scripts chuẩn bị dataset fine-tune cho 2 model (coder + instruct).

## Cấu trúc

```
data_prep/
├── __init__.py
├── _common.py                    # Shared utilities (load data, format, etc.)
├── prepare_coder_dataset.py      # Tạo coder.jsonl (Z3 + SymPy training data)
└── prepare_instruct_dataset.py   # Tạo instruct.jsonl (ExactResponse training data)
```

## Chi tiết

### _common.py

- Load `electro_dataset.jsonl` và `electro_sympy_dataset.jsonl` từ `data/collected/`.
- Load BTC dataset từ `data/EXACT2026_dataset_2026-05-15/`.
- Các hàm format chung: tạo chat messages, normalize output.

### prepare_coder_dataset.py

- **Output**: `data/finetune/coder.jsonl`
- **Nội dung**: Các cặp (prompt, code) để fine-tune model sinh Z3/SymPy code.
- **Nguồn**: BTC Logic dataset (Z3) + BTC Physics dataset (SymPy).

### prepare_instruct_dataset.py

- **Output**: `data/finetune/instruct.jsonl`
- **Nội dung**: Các cặp (prompt, ExactResponse JSON) để fine-tune model sinh explanation.
- **Nguồn**: BTC dataset + electro dataset.

## Cách chạy

```bash
cd "Exact 2026"
venv\Scripts\python.exe -m scripts.data_prep.prepare_coder_dataset
venv\Scripts\python.exe -m scripts.data_prep.prepare_instruct_dataset
```
