# src/utils/

Tiện ích dùng chung cho toàn bộ project.

## Cấu trúc

```
utils/
├── __init__.py
├── logger.py            # Setup logging từ config/logging.yaml
├── code_extract.py      # Trích xuất Python code từ LLM output
├── safe_python.py       # Validate + chạy generated code với giới hạn
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

### safe_python.py

- Validate AST và allowlist import trước khi chạy code do model sinh.
- Chạy Python isolated, bỏ secrets khỏi environment, giới hạn output và timeout.
- Trên Linux còn giới hạn CPU, memory, file size và process count.
- Dừng toàn process group khi request bị cancel.
