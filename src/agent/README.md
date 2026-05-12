# Agent Module — EXACT 2026

Thành phần trung tâm điều phối toàn bộ pipeline xử lý bài toán Logic và Vật lý sử dụng **LangGraph**.

## 🏗️ Kiến trúc Pipeline

Pipeline được thiết kế theo dạng đồ thị có trạng thái (Stateful Graph) với cơ chế chạy song song (Parallel Execution) để tối ưu hóa độ chính xác và tốc độ.

### Các luồng xử lý chính:

1.  **Phân loại (Classification)**:
    *   Dựa trên cấu trúc input (có giả thiết hay không) để xác định là `logic` hay `physics`.
2.  **Nhánh Logic (Z3 Solver)**:
    *   **Formalizer**: LLM dịch bài toán sang mã Python sử dụng thư viện Z3.
    *   **Solver**: Thực thi mã code để tìm ra lời giải ký hiệu chính xác.
    *   **Direct (Parallel)**: LLM suy luận trực tiếp để làm phương án dự phòng (fallback).
3.  **Nhánh Vật lý (SymPy Solver)**:
    *   **RAG**: Truy xuất công thức vật lý liên quan từ Vector Database.
    *   **Formalizer**: LLM dịch bài toán và công thức sang mã Python SymPy.
    *   **Solver**: Thực thi mã để tính toán kết quả số học kèm đơn vị SI.
    *   **Direct (Parallel)**: LLM tự giải toán dựa trên kiến thức và ngữ cảnh RAG.

## 📁 Cấu trúc thư mục

*   `graph.py`: Định nghĩa cấu trúc đồ thị, các cạnh (edges) và điểm bắt đầu.
*   `state.py`: Định nghĩa `AgentState` - Nguồn sự thật duy nhất (SSoT) được chia sẻ giữa các node.
*   `nodes/`: Chứa logic xử lý chi tiết cho từng bước:
    *   `classifier.py`: Logic phân loại và router.
    *   `logic_node.py`: Xử lý bài toán Logic quy chế.
    *   `physics_node.py`: Xử lý bài toán Vật lý điện học.

## 🔄 Luồng dữ liệu (State Management)

Toàn bộ pipeline sử dụng `AgentState` để truyền thông tin. Các node được thiết kế để chỉ trả về phần dữ liệu mà chúng cập nhật, giúp tránh xung đột khi thực thi song song.

```python
class AgentState(TypedDict):
    question: str           # Câu hỏi đầu vào
    premises: list[str]     # Giả thiết (nếu có)
    task_type: str          # Loại bài toán (logic/physics)
    intermediate_answer: ... # Kết quả từ bộ giải code (Z3/SymPy)
    fallback_answer: ...     # Kết quả suy luận trực tiếp từ LLM
    final_answer: ...        # Đáp án cuối cùng sau khi tổng hợp
```

## 🚀 Cách sử dụng

Sử dụng hàm `run_pipeline` trong `graph.py` để thực thi:

```python
from src.agent.graph import run_pipeline

result = run_pipeline(
    question="Câu hỏi của bạn ở đây",
    premises=["Giả thiết 1", "Giả thiết 2"]  # (Tùy chọn)
)

print(result["answer"])
```

## 🛠️ Yêu cầu cài đặt bổ sung

Để bộ giải mã (Solver) hoạt động, cần cài đặt các thư viện tính toán:
*   `z3-solver`: Dùng cho nhánh Logic.
*   `sympy`: Dùng cho nhánh Vật lý.
