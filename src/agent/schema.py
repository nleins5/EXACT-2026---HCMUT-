from typing import List, Optional
from pydantic import BaseModel, Field

class ExactResponse(BaseModel):
    """
    Cấu trúc phản hồi chính thức cho cuộc thi EXACT 2026.
    Gồm các trường bắt buộc và khuyến khích để tăng chiều sâu lập luận.
    """
    answer: str = Field(..., description="Đáp án chính xác (A, B, C, Yes, No, hoặc giá trị số)")
    explanation: str = Field(..., description="Lập luận giải thích chi tiết bằng ngôn ngữ tự nhiên")
    
    # Các trường bổ sung (khuyến khích)
    fol: Optional[str] = Field(None, description="Dịch bài toán sang logic bậc nhất (First-Order Logic)")
    cot: Optional[List[str]] = Field(None, description="Các bước suy luận Chain-of-Thought")
    premises: Optional[List[str]] = Field(None, description="Danh sách các giả thiết được sử dụng")
    confidence: Optional[float] = Field(None, description="Mức độ tự tin của model (0.0 - 1.0)")
