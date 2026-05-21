# src/api/ (API Specification & Guidelines)

Tài liệu này là **Đặc tả API (API Specification)** chi tiết dành cho người phát triển API phục vụ hệ thống EXACT-2026. API này sẽ là cổng giao tiếp duy nhất giữa hệ thống chấm điểm của Ban Tổ Chức (BTC) và mô hình Agentic AI của chúng ta.

Tất cả các định nghĩa dưới đây tuân thủ nghiêm ngặt theo **EXACT 2026 Kick-off Workshop Slides** và **Official Q&A Document**.

---

## 1. Yêu Cầu Kỹ Thuật (Theo Quy Định BTC EXACT 2026)

Người code API CẦN đặc biệt chú ý các ràng buộc sau để tránh bị loại (Disqualified) hoặc 0 điểm:

1. **Hard Cap 60s/Request:** 
   - BTC quy định timeout là 60 giây. Nếu request vượt quá 60s, hệ thống chấm điểm sẽ đánh lỗi (failed answer).
   - **Bắt buộc:** Phải bọc việc gọi pipeline bằng `asyncio.wait_for` (timeout ~58 giây). Nếu bị TimeoutError, trả về Fallback Response (xem Phần 5).
2. **Unified Endpoint (Single API):**
   - Chỉ sử dụng MỘT endpoint (e.g., `POST /predict`) để xử lý cả 2 loại bài toán (Type 1 Logic và Type 2 Physics).
   - Dataset test của BTC sẽ gộp chung 2 loại thành một luồng (unified stream).
3. **Single-resident LLM (Quan trọng về bộ nhớ):**
   - Chỉ được load 1 LLM (<= 8B) lên GPU tại một thời điểm.
   - API Server (FastAPI) phải chạy với **duy nhất 1 worker** (`uvicorn --workers 1`). Không dùng multi-processing cho uvicorn để tránh mỗi worker tự spawn `llama-server` gây tràn VRAM.
4. **OpenAI-compatible Serving:**
   - Việc serve model phải thông qua `llama-server` (hoặc vLLM) expose OpenAI API (`/v1/chat/completions`). Pipeline hiện tại đã quản lý việc này qua `LlamaServerSupervisor`. API chỉ cần init supervisor ở vòng đời lifespan.

---

## 2. API Endpoints Đặc Tả

### 2.1. `POST /predict` (Endpoint Chính)

Nhận request chứa câu hỏi từ hệ thống chấm điểm BTC và trả về kết quả.

**Request Body (JSON):**
Dựa trên Slide 32, BTC sẽ gửi data khác nhau tùy loại câu hỏi, nhưng quy chung về 1 schema:
```json
{
  "question": "Nội dung câu hỏi (Logic hoặc Vật lý)",
  "premises-NL": [
    "Giả thiết 1 (chỉ có trong bài Type 1 Logic)",
    "Giả thiết 2"
  ]
}
```
*Lưu ý:* `premises-NL` là optional. Bài Type 2 (Physics) thường sẽ không có field này hoặc là mảng rỗng. Classifier node trong LangGraph sẽ tự động route dựa trên việc có `premises` hay không.

**Response Body (JSON):**
Theo Slide 33, `answer` và `explanation` là bắt buộc. Các field còn lại là tùy chọn nhưng được khuyến khích để lấy điểm Reasoning Depth (P3).
```json
{
  "answer": "Yes/No/Unknown hoặc đáp án số kèm unit (vd: 20.0 Ω)",
  "explanation": "Giải thích chi tiết bằng ngôn ngữ tự nhiên",
  "fol": "First-Order Logic derivation (nếu có, cho Type 1)",
  "cot": [
    "Bước 1: Tính điện trở tương đương...",
    "Bước 2: Áp dụng định luật Ohm..."
  ],
  "premises": [
    "Ohm's law: V = IR"
  ],
  "confidence": 0.95
}
```

### 2.2. `GET /health` (Endpoint Monitor)

Dùng để check trạng thái server trước Public Test Day.
**Response:**
```json
{
  "status": "ok",
  "supervisor_running": true,
  "uptime": 3600
}
```

---

## 3. Hướng Dẫn Tích Hợp LangGraph Pipeline

Code API **không cần** biết cách gọi `llama-server` hay setup LangChain. Chỉ cần mapping Request vào StateGraph.

**Ví dụ cấu trúc `src/api/routes.py` và `src/app.py`:**

```python
# Mẫu Lifespan trong src/app.py
from fastapi import FastAPI
from src.agent.llm.factory import LLMFactory
from src.agent.llm.server_supervisor import LlamaServerSupervisor
from src.agent.graph import get_graph

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
```

**Mẫu gọi Graph trong Route:**
```python
# Mẫu gọi trong src/api/routes.py
import asyncio
from fastapi import APIRouter, HTTPException

router = APIRouter()
graph = get_graph()

@router.post("/predict")
async def predict(request: RequestSchema):
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

## 4. Xử Lý Fallback (Crash/Timeout)

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

## 5. Deployment Checklist (Dành Cho API Dev)

- [ ] API phải handle được tải (mặc dù chỉ 1 worker nhưng có thể dùng async).
- [ ] Tham số `host` và `port` linh hoạt thông qua biến môi trường hoặc CLI (để deploy lên ngrok/local/cloud tùy ý đồ test).
- [ ] Chắc chắn file `config/setting.yaml` đã cài đặt đường dẫn GGUF đúng trước khi khởi động FastAPI.
- [ ] In logs chi tiết (Question ID nếu có, thời gian response) bằng `src.utils.logger` để dễ debug lúc thi live.
