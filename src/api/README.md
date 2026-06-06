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
  "ready": true,
  "busy": false,
  "supervisor_running": true,
  "active_role": "coder",
  "startup_error": null,
  "uptime": 3600
}
```

`status` chỉ là `ok` khi model server thực sự đang chạy và sẵn sàng nhận request.

---

## 4. Hướng Dẫn Tích Hợp LangGraph Pipeline

`src/api/app.py` khởi tạo một `LlamaServerSupervisor`, warm model `coder`, rồi
chỉ báo `health.status=ok` khi model process thực sự sẵn sàng.

`src/api/routes/predict.py` dùng một request gate để serialize toàn pipeline.
Mỗi request có deadline và `threading.Event` cancellation. Khi hết budget, route
kill active model process, dừng solver child process, giữ gate cho tới khi
background pipeline thoát, rồi trả response fallback hợp lệ.

`src/agent/graph.py` cũng giữ pipeline-level lock để các lời gọi trực tiếp ngoài
HTTP không thể swap model đồng thời.

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

- [ ] Chạy đúng 1 Uvicorn worker; request được queue/serialize vì chỉ một model được resident.
- [ ] Tham số `host` và `port` linh hoạt thông qua biến môi trường hoặc CLI (để deploy lên ngrok/local/cloud tùy ý đồ test).
- [ ] Chắc chắn file `config/setting.yaml` đã cài đặt đường dẫn GGUF đúng trước khi khởi động FastAPI.
- [ ] In logs chi tiết (Question ID nếu có, thời gian response) bằng `src.utils.logger` để dễ debug lúc thi live.
- [ ] Giữ `EXACT_REQUEST_BUDGET_SECONDS <= 58` để chừa biên an toàn dưới hard cap 60 giây/request của Q&A Q13.
- [ ] Kiểm tra `GET /v1/models` trước khi nộp; endpoint này phục vụ audit Q5/Q14 về self-hosted OpenAI-compatible serving.
