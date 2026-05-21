# src/agent/prompts/

Prompt templates cho các node trong pipeline.

## Cấu trúc

```
prompts/
├── __init__.py                 # Re-export tất cả prompts
├── logic_formalizer.py         # LOGIC_FORMALIZER_PROMPT (sinh Z3 code)
├── logic_explanation.py        # LOGIC_OUTPUT_PROMPT, LOGIC_OUTPUT_ERROR_PROMPT
├── physics_formalizer.py       # PHYSICS_FORMALIZER_PROMPT (sinh SymPy code)
└── physics_explanation.py      # PHYSICS_OUTPUT_PROMPT, PHYSICS_OUTPUT_ERROR_PROMPT
```

## Quy tắc đặt tên

- `*_FORMALIZER_PROMPT`: prompt cho Coder model (sinh code).
- `*_OUTPUT_PROMPT`: prompt cho Instruct model khi code chạy thành công.
- `*_OUTPUT_ERROR_PROMPT`: prompt cho Instruct model khi code lỗi (fallback).

## Hai nhánh prompt (explanation nodes)

Mỗi explanation node có 2 prompt:
1. **SUCCESS** (`*_OUTPUT_PROMPT`): code chạy OK → tin kết quả solver, sinh ExactResponse.
2. **ERROR** (`*_OUTPUT_ERROR_PROMPT`): code lỗi → đọc code + error làm hint, tự suy luận.

## Output format

Tất cả explanation prompts yêu cầu output JSON theo schema `ExactResponse`:
```json
{
  "answer": "Yes/No/Unknown hoặc giá trị số",
  "explanation": "Lập luận chi tiết",
  "fol": "First-Order Logic (optional)",
  "cot": ["Bước 1", "Bước 2"],
  "premises": ["Giả thiết 1", "Giả thiết 2"],
  "confidence": 0.9
}
```
