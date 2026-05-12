from typing import TypedDict, Literal

class FinalAnswer(TypedDict):
    """Cấu trúc kết quả cuối cùng sau khi đã được định dạng."""
    answer: str        # Đáp án rút gọn (ví dụ: A, B, 10.5)
    reasoning: str     # Lập luận giải thích chi tiết
    final_output: str  # Chuỗi văn bản đầy đủ để hiển thị cho người dùng

class FallbackAnswer(TypedDict):
    """Kết quả dự phòng từ LLM khi bộ giải (solver) gặp lỗi."""
    answer: str
    reasoning: str
    final_output: str

class IntermediateAnswer(TypedDict):
    """Các kết quả trung gian trong quá trình xử lý của các Node."""
    context_rag: str        # Nội dung truy xuất được từ RAG (nếu có)
    context_code: str       # Ngữ cảnh bổ sung cho việc sinh mã
    generated_code: str     # Mã Python (Z3 cho logic, SymPy cho vật lý) được LLM sinh ra
    code_output: str        # Kết quả thực thi của mã trên (stdout/stderr)
    reasoning: str          # Suy luận trung gian (nếu cần)
    final_output: str

class AgentState(TypedDict):
    """
    Trạng thái (State) được chia sẻ giữa tất cả các Node trong pipeline LangGraph.
    Đây là nguồn sự thật duy nhất (Single Source of Truth) cho mỗi lượt chạy.
    """

    # Input (Đầu vào từ người dùng)
    question: str           # Câu hỏi gốc
    premises: list[str]     # Các giả thiết bài toán logic (Type 1 - NL premises)
    collection_name: str    # Tên bộ dữ liệu RAG cần truy vấn
    
    # Classification (Phân loại bài toán)
    task_type: Literal["logic", "physics", "unknown"]
    
    # Intermediate results (Kết quả trung gian)
    context: str            # Ngữ cảnh thô (ví dụ: text từ văn bản quy chế)
    intermediate_answer: IntermediateAnswer
    
    # Final output (Kết quả đầu ra cuối cùng)
    final_answer: FinalAnswer
    
    # Fallback (Dự phòng: LLM suy luận trực tiếp không dùng solver)
    fallback_answer: FallbackAnswer
    
    # Control & Debug (Điều khiển và Gỡ lỗi)
    error: str              # Ghi lại thông báo lỗi nếu có node nào thất bại
