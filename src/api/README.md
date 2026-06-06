# src/api/ (API Specification & Guidelines)

Tài liệu này là **Đặc tả API (API Specification)** chi tiết dành cho người phát triển API phục vụ hệ thống EXACT-2026. API này sẽ là cổng giao tiếp duy nhất giữa hệ thống chấm điểm của Ban Tổ Chức (BTC) và mô hình Agentic AI của chúng ta.

Tất cả các định nghĩa dưới đây tuân thủ nghiêm ngặt theo **EXACT 2026 Kick-off Workshop Slides** và **Official Q&A Document**.

---

## 1. Yêu Cầu Kỹ Thuật (Theo Quy Định BTC EXACT 2026)

Người code API CẦN đặc biệt chú ý các ràng buộc sau để tránh bị loại (Disqualified) hoặc 0 điểm:

1. **Unified Endpoint (Single API):**
   - Chỉ sử dụng MỘT endpoint (e.g., `POST /predict`) để xử lý cả 2 loại bài toán (Type 1 Logic và Type 2 Physics).
   - Dataset test của BTC sẽ gộp chung 2 loại thành một luồng (unified stream).
2. **Single-resident LLM (Quan trọng về bộ nhớ):**
   - Chỉ được load 1 LLM (<= 8B) lên GPU tại một thời điểm.
   - API Server (FastAPI) phải chạy với **duy nhất 1 worker** (`uvicorn --workers 1`). Không dùng multi-processing cho uvicorn để tránh mỗi worker tự spawn `llama-server` gây tràn VRAM.
3. **OpenAI-compatible Serving:**
   - Việc serve model phải thông qua `llama-server` (hoặc vLLM) expose OpenAI API (`/v1/chat/completions`). Pipeline hiện tại đã quản lý việc này qua `LlamaServerSupervisor`. API chỉ cần init supervisor ở vòng đời lifespan.

---

## 2. Cấu Trúc Thư Mục Đề Xuất (Folder Structure)

Để code gọn gàng và dễ bảo trì, vui lòng tổ chức thư mục `src/api/` theo cấu trúc sau. Các endpoints và schemas phải được đặt trong thư mục riêng:

```text
src/api/
├── __init__.py
├── app.py              # Main FastAPI application, cấu hình CORS, middlewares & lifespan
├── routes/             # Thư mục chứa logic của các endpoints
│   ├── __init__.py
│   ├── predict.py      # Định nghĩa POST /predict (gọi LangGraph pipeline)
│   └── health.py       # Định nghĩa GET /health
└── schemas/            # Thư mục chứa Pydantic models (Data Validation)
    ├── __init__.py
    ├── request.py      # Định nghĩa RequestSchema
    └── response.py     # Định nghĩa ResponseSchema (ExactResponse)
```

---

## 3. API Endpoints Đặc Tả

### 3.1. `POST /predict` (Endpoint Chính)

Nhận request chứa câu hỏi từ hệ thống chấm điểm BTC và trả về kết quả. (Nên code trong `routes/predict.py`).

**Request Body (JSON):**
Dựa trên Slide 32, BTC sẽ gửi data khác nhau tùy loại câu hỏi, nhưng quy chung về 1 schema (định nghĩa trong `schemas/request.py`):

```json
{
  "question": "Nội dung câu hỏi (Logic hoặc Vật lý)",
  "task_type": "logic",
  "premises-NL": ["Giả thiết 1 (chỉ có trong bài Type 1 Logic)", "Giả thiết 2"]
}
```

_Lưu ý:_ Theo Official Q&A Q18, payload test có thể gửi field chỉ định query type. API hỗ trợ các biến thể `task_type`, `query_type`, `problem_type`, `type`, `task-type`, `query-type` với giá trị như `logic`, `physics`, `type-1`, `type-2`, `1`, `2`. Nếu field này không có, classifier vẫn route dựa trên `premises-NL`: có premises → logic, rỗng → physics.

**Response Body (JSON):**
Theo Slide 33, `answer` và `explanation` là bắt buộc. Các field còn lại là tùy chọn nhưng được khuyến khích để lấy điểm Reasoning Depth (P3). (Định nghĩa trong `schemas/response.py`).

```json
{
  "answer": "Yes/No/Unknown hoặc đáp án số kèm unit (vd: 20.0 Ω)",
  "explanation": "Giải thích chi tiết bằng ngôn ngữ tự nhiên",
  "fol": "First-Order Logic derivation (nếu có, cho Type 1)",
  "cot": [
    "Bước 1: Tính điện trở tương đương...",
    "Bước 2: Áp dụng định luật Ohm..."
  ],
  "premises": ["Ohm's law: V = IR"],
  "confidence": 0.95
}
```

### 3.2. `GET /health` (Endpoint Monitor)

Dùng để check trạng thái server trước Public Test Day. (Nên code trong `routes/health.py`).
**Response:**

```json
{
  "status": "ok",
  "supervisor_running": true,
  "uptime": 3600
}
```

---

## 4. Hướng Dẫn Tích Hợp LangGraph Pipeline

Code API **không cần** biết cách gọi `llama-server` hay setup LangChain. Chỉ cần mapping Request vào StateGraph.

**Ví dụ cấu trúc `src/api/app.py`:**

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.agent.llm.factory import LLMFactory
from src.agent.llm.server_supervisor import LlamaServerSupervisor
from src.api.routes import predict, health

supervisor = LlamaServerSupervisor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    LLMFactory.init(supervisor)
    # Warmup instruct model (Tùy chọn)
    supervisor.swap_to("instruct")
    yield
    # Shutdown
    supervisor.shutdown()

app = FastAPI(lifespan=lifespan)
app.include_router(predict.router)
app.include_router(health.router)
```

**Ví dụ gọi Graph trong Route (`src/api/routes/predict.py`):**

```python
import asyncio
from fastapi import APIRouter
from src.api.schemas.request import RequestSchema
from src.api.schemas.response import ResponseSchema
from src.agent.graph import get_graph

router = APIRouter()
graph = get_graph()

@router.post("/predict", response_model=ResponseSchema)
async def predict_endpoint(request: RequestSchema):
    try:
        # Giới hạn 55s để an toàn (BTC cap 60s)
        result = await asyncio.wait_for(
            asyncio.to_thread(
                graph.invoke,
                {
                    "question": request.question,
                    "premises": request.dict().get("premises-NL", [])
                }
            ),
            timeout=55.0
        )
        final = result.get("final_answer", {})
        return ResponseSchema(**final)

    except asyncio.TimeoutError:
        return fallback_response()
    except Exception as e:
        return fallback_response(error=str(e))
```

---

## 5. Xử Lý Fallback (Crash/Timeout)

**TUYỆT ĐỐI KHÔNG** trả về HTTP 500 trắng hoặc văng lỗi cho hệ thống BTC. Kể cả khi crash hoặc hết 55 giây, API phải trả về HTTP 200 OK với một đáp án rỗng hợp lệ để bộ chấm tự động không bị kẹt.

**Mẫu Fallback Response:**

```json
{
  "answer": "Unknown",
  "explanation": "Hệ thống gặp lỗi nội bộ hoặc quá thời gian xử lý 60s (Timeout).",
  "fol": "",
  "cot": [],
  "premises": [],
  "confidence": 0.0
}
```

---

## 6. Deployment Checklist (Dành Cho API Dev)

- [ ] API phải handle được tải (mặc dù chỉ 1 worker nhưng có thể dùng async).
- [ ] Tham số `host` và `port` linh hoạt thông qua biến môi trường hoặc CLI (để deploy lên ngrok/local/cloud tùy ý đồ test).
- [ ] Chắc chắn file `config/setting.yaml` đã cài đặt đường dẫn GGUF đúng trước khi khởi động FastAPI.
- [ ] In logs chi tiết (Question ID nếu có, thời gian response) bằng `src.utils.logger` để dễ debug lúc thi live.
- [ ] Giữ `EXACT_REQUEST_BUDGET_SECONDS <= 58` để chừa biên an toàn dưới hard cap 60 giây/request của Q&A Q13.
- [ ] Kiểm tra `GET /v1/models` trước khi nộp; endpoint này phục vụ audit Q5/Q14 về self-hosted OpenAI-compatible serving.
