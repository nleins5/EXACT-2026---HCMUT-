# src/utils/

Tiện ích dùng chung cho toàn bộ project.

## Cấu trúc

```
utils/
├── __init__.py
├── logger.py            # Setup logging từ config/logging.yaml
├── code_extract.py      # Trích xuất Python code từ LLM output
└── z3_output_parser.py  # Parse output Z3 solver → True/False/Unknown
```

## Chi tiết

### logger.py

- Đọc `config/logging.yaml`, tạo logger tên `"exact"`.
- Tự động chạy `setup_logging()` khi import.
- Dùng: `from src.utils.logger import logger`

### code_extract.py

- `extract_python_code(text) → str`: trích code Python từ output LLM.
- Xử lý: `<think>` blocks, fenced code blocks (đóng/mở), plain code.
- Trả `""` nếu không tìm thấy code hợp lệ → downstream set `code_error=True`.

### z3_output_parser.py

- `parse_z3_output(raw_output) → "True" | "False" | "Unknown"`.
- Xử lý nhiều format output Z3 (Predicted, Expected, multi-conclusion, majority vote).
- Dùng trong `logic_solver` node.
