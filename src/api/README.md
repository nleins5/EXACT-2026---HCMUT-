# src/api/

Thư mục này chứa code HTTP API (FastAPI) cho EXACT-2026.

## Cấu trúc

```
src/api/
├── README.md               # File này
├── __init__.py             # Export: router
├── routes.py               # POST /predict + GET /health
└── schemas.py              # PredictRequest / PredictResponse / HealthResponse
```

## Mục đích

Thư mục `src/api/` chứa:
- **Endpoints**: `/predict` (Type 1 + Type 2), `/health`
- **Schemas**: Pydantic models cho request/response

## Chi tiết từng module

### routes.py

**Endpoints**:

| Endpoint | Method | Mục đích |
|----------|--------|----------|
| `/predict` | POST | Xử lý request Type 1 (logic) + Type 2 (physics) |
| `/health` | GET | Liveness + readiness probe |

**POST /predict**:

**Request**:
```json
{
  "question": "Is it true that all birds can fly?",
  "premises-NL": ["Birds have wings", "Things with wings can fly"]
}
```

**Response**:
```json
{
  "answer": "No",
  "explanation": "Not all birds can fly...",
  "fol": "∃x (Bird(x) ∧ ¬CanFly(x))",
  "cot": ["Identify premise", "Check exceptions", "Conclude"],
  "premises": ["Premise 1: ...", "Premise 2: ..."],
  "confidence": 0.85
}
```

**Timeout**: 60s (hard cap BTC Q13)

**Status codes**:
- `200`: Success
- `504`: Timeout (>budget)
- `500`: Pipeline error

**Logic**:
1. Parse `premises-NL` (alias cho `premises_nl`)
2. Run `run_pipeline(question, premises)` trong executor
3. Apply `asyncio.wait_for(..., timeout=budget)`
4. Return `PredictResponse`

### schemas.py

**Pydantic models**:

| Model | Mục đích |
|-------|----------|
| `PredictRequest` | Request body cho `/predict` |
| `PredictResponse` | Response body cho `/predict` |
| `HealthResponse` | Response body cho `/health` |

**PredictRequest**:
```python
class PredictRequest(BaseModel):
    question: str
    premises_nl: list[str] = Field(default=None, alias="premises-NL")
```

**PredictResponse**:
```python
class PredictResponse(BaseModel):
    answer: str
    explanation: str
    fol: list[str] | None = None
    cot: list[str] | None = None
    premises: list[str] | None = None
    confidence: float | None = None
```

**HealthResponse**:
```python
class HealthResponse(BaseModel):
    status: Literal["ok", "warming"]
    model_loaded: bool
    elapsed_warmup_seconds: float | None = None
```

## Workflow

```
Client → FastAPI /predict
  → asyncio.wait_for(run_pipeline, timeout=60s)
    → LangGraph pipeline
  → PredictResponse
```

## Yêu cầu

```powershell
pip install fastapi uvicorn langchain-openai
```

## Environment variables

- `EXACT_REQUEST_BUDGET_SECONDS`: Timeout cho `/predict` (mặc định 60s)
