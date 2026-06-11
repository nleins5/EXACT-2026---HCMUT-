from typing import Literal, NotRequired, TypedDict


class FinalAnswer(TypedDict):
    answer: str
    explanation: str
    fol: str             # Optional — used to build reasoning block
    cot: list[str]       # Optional — used to build reasoning block
    premises: list[str]  # Optional legacy — text of used premises
    premises_used: list[int]  # 0-based indices of premises actually used
    unit: str            # ASCII unit for Type 2; empty for Type 1
    confidence: float    # Optional internal metric


class IntermediateAnswer(TypedDict):
    context_rag: str
    context_code: str
    generated_code: str
    code_output: str
    code_error: bool
    error_message: str
    retry_error_feedback: NotRequired[str]
    reasoning: str
    final_output: str


class AgentState(TypedDict):
    """Shared state across all LangGraph nodes."""

    question: str
    premises: list[str]
    options: list[str]            # Choice set from evaluation server
    collection_name: str

    task_type: Literal["logic", "physics"]
    requested_task_type: NotRequired[Literal["logic", "physics"] | None]

    context: str
    intermediate_answer: IntermediateAnswer

    final_answer: FinalAnswer

    error: str

    retry_count: NotRequired[int]
