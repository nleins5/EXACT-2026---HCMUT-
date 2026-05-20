# src/distillation/

Thư mục này chứa code offline cho pipeline distillation (build Physics KB cho RAG).

## Cấu trúc

```
src/distillation/
├── README.md               # File này
├── __init__.py
├── prompts.py              # Prompts cho teacher LLM (extract/generate mode)
├── schema.py               # KBRecord dataclass
└── teacher_client.py       # GeminiTeacherClient async wrapper
```

## Mục đích

Thư mục `src/distillation/` chứa:
- **Schema**: `KBRecord` - cấu trúc dữ liệu cho Physics KB
- **Prompts**: Prompts cho teacher LLM (Gemini) trích xuất KB
- **Teacher Client**: Async wrapper cho Google Gemini SDK

**Ghi chú**: Đây là code **offline**, KHÔNG được gọi trong runtime pipeline.

## Chi tiết từng module

### schema.py

**KBRecord** - Dataclass cho Physics KB record:

```python
@dataclass
class KBRecord:
    id: str                          # vd "btc_pb_42"
    source: str                      # "btc_physics" | "electro" | "physics_formulae"
    problem: str                     # statement gốc
    topic: str                       # "electrostatics" | "electric_circuits" | "other"
    formulas: list[str]              # plain math
    symbols: dict[str, str]          # var → mô tả + đơn vị
    sympy_code: str                  # code SymPy in `FINAL_ANSWER: ...`
    answer: str                      # "20 Ohm"
    derivation: str                  # 1-3 câu

    # Verify metadata (set by verify_kb.py)
    verified: bool | None
    exec_output: str
    exec_error: str

    # Cost tracking
    teacher_model: str
    input_tokens: int
    output_tokens: int
```

### prompts.py

**Prompts cho teacher LLM**:

- `EXTRACT_SYSTEM_PROMPT`: Trích xuất từ BTC CoT (mặc định)
- `GENERATE_SYSTEM_PROMPT`: Sinh từ đầu (fallback khi không có CoT)

**Extract mode**: Teacher chỉ TRÍCH XUẤT công thức + code từ CoT có sẵn, không sinh từ đầu → rẻ hơn + ít hallucination.

### teacher_client.py

**GeminiTeacherClient** - Async wrapper cho Google Gemini SDK:

- Model mặc định: `gemini-2.5-flash-lite` (rẻ nhất)
- `response_mime_type=application/json` ép output JSON pure
- Retry exponential backoff cho rate-limit / 5xx
- API key từ env var `GOOGLE_API_KEY`

**Method**:
- `distill_one(record_id, source, problem, hint, answer, unit)` → `KBRecord`

## Workflow

```
BTC/Electro → distill_physics.py → teacher_client → physics_kb.raw.jsonl
physics_kb.raw.jsonl → verify_kb.py → physics_kb.verified.jsonl
PhysicsFormulae → fetch_physics_formulae.py → physics_kb.from_pf.jsonl
```

## Yêu cầu

```powershell
pip install google-generativeai
```

## Environment variables

- `GOOGLE_API_KEY`: API key cho Google Gemini
