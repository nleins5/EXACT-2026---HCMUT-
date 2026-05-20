# scripts/distill/

Thư mục này chứa pipeline distillation để xây dựng Physics Knowledge Base (KB) cho RAG.

## Cấu trúc

```
scripts/distill/
├── README.md               # File này
├── __init__.py             # Export: KBRecord, GeminiTeacherClient, prompts
├── prompts.py              # Prompts cho teacher LLM (extract/generate mode)
├── schema.py               # KBRecord dataclass
├── teacher_client.py       # GeminiTeacherClient async wrapper
├── distill_physics.py      # Teacher LLM → physics_kb.raw.jsonl
├── verify_kb.py            # Exec SymPy → mark verified
└── fetch_physics_formulae.py # Pull công thức từ PhysicsFormulae GitHub
```

## Mục đích

Thư mục `scripts/distill/` xây dựng Physics KB cho `physics_rag_node` bằng cách:

1. **fetch_physics_formulae**: Crawl PhysicsFormulae GitHub → 22 formulas + 6 constants
2. **distill_physics**: Teacher LLM (Gemini) trích xuất KB từ BTC/Electro
3. **verify_kb**: Exec SymPy code → mark `verified=true/false`

KB cuối cùng chỉ thuộc 2 topic theo scope EXACT 2026: `electrostatics` + `electric_circuits`.

## Chi tiết từng module

### prompts.py

**Prompts cho teacher LLM**:

- `EXTRACT_SYSTEM_PROMPT`: Trích xuất từ BTC CoT (mặc định)
- `GENERATE_SYSTEM_PROMPT`: Sinh từ đầu (fallback khi không có CoT)

**Extract mode**: Teacher chỉ TRÍCH XUẤT công thức + code từ CoT có sẵn, không sinh từ đầu → rẻ hơn + ít hallucination.

**Prompts bằng tiếng Anh** vì:
- Dataset BTC có prompt tiếng Anh
- Model fine-tune học từ prompt tiếng Anh
- Output JSON phải match format fine-tune dataset

### schema.py

**KBRecord** - Dataclass cho Physics KB record:

```python
@dataclass
class KBRecord:
    id: str                          # uid: "btc_pb_42", "electro_023"
    source: str                      # "btc_physics" | "electro" | "physics_formulae"
    problem: str                     # Problem statement (English)
    topic: str                       # "electrostatics" | "electric_circuits" | "other"
    formulas: list[str]              # plain math formulas
    symbols: dict[str, str]          # var -> description + unit
    sympy_code: str                  # runnable SymPy code ending with print('FINAL_ANSWER: ...')
    answer: str                      # final answer with unit (e.g., "20 Ohm")
    derivation: str                  # brief: 1-3 sentences explaining the reasoning
    verified: bool | None
    exec_output: str
    exec_error: str
    teacher_model: str
    input_tokens: int
    output_tokens: int
```

### teacher_client.py

**GeminiTeacherClient** - Async wrapper cho Google Gemini SDK:

- Model mặc định: `gemini-2.5-flash-lite` (rẻ nhất)
- `response_mime_type=application/json` ép output JSON pure
- Retry exponential backoff cho rate-limit / 5xx
- API key từ env var `GOOGLE_API_KEY`

**Method**:
- `distill_one(record_id, source, problem, hint, answer, unit)` → `KBRecord`

### distill_physics.py

**Async pipeline** gọi teacher LLM:

**Usage**:
```powershell
.\venv\Scripts\python.exe -m scripts.distill.distill_physics --source all
# --source btc | electro | all
# --limit 10                # thử 10 record
# --concurrency 4           # giảm khi rate-limit
# --dry-run                 # chỉ liệt kê ID, không gọi LLM
```

**Resumable**: Đọc file output cũ, skip ID đã có.

**Output**: Append vào `physics_kb.raw.jsonl` + `cost_log.jsonl`.

### verify_kb.py

**Exec SymPy code** từng record:

**Usage**:
```powershell
.\venv\Scripts\python.exe -m scripts.distill.verify_kb
# --in custom.raw.jsonl
# --out custom.verified.jsonl
# --timeout 5
```

**Timeout**: 10s mặc định.

**Output**: `physics_kb.verified.jsonl` với `verified=true/false`.

### fetch_physics_formulae.py

**Crawl PhysicsFormulae GitHub**:

**Usage**:
```powershell
.\venv\Scripts\python.exe -m scripts.distill.fetch_physics_formulae --include-constants
# --dry-run
# --force-download
```

**Filter**: Giữ `Classical Electromagnetism` + `Electrical Circuits`, bỏ mechanics/thermodynamics/optics/quantum.

**Constants**: `epsilon_0`, `mu_0`, `e`, `k_e`, `m_e`, `m_p`.

**Output**: `physics_kb.from_pf.jsonl` (28 records, verified=true mặc định).

## Workflow đầy đủ

```powershell
# 1. Pull công thức + 6 hằng từ PhysicsFormulae
.\venv\Scripts\python.exe -m scripts.distill.fetch_physics_formulae --include-constants

# 2. Distill thêm từ BTC + electro
$env:GOOGLE_API_KEY = "<your-key>"
.\venv\Scripts\python.exe -m scripts.distill.distill_physics --source all

# 3. Verify SymPy code
.\venv\Scripts\python.exe -m scripts.distill.verify_kb

# 4. Build 2 collection Qdrant
.\venv\Scripts\python.exe -m scripts.rag.build_physics_index --rebuild
```

## Chi phí

- **Gemini Flash Lite free tier**: 0 USD (qua đêm với 15 RPM rate-limit)
- **Trả phí + Batch API**: ~0.18 USD cho toàn bộ 1,594 bài (BTC 1,352 + electro 242)

## Yêu cầu

```powershell
pip install google-generativeai
```

## Environment variables

- `GOOGLE_API_KEY`: API key cho Google Gemini
