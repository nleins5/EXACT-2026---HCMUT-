from typing import Literal, NotRequired, TypedDict

class FinalAnswer(TypedDict):
    answer: str
    explanation: str
    fol: str             # Optional
    cot: list[str]       # Optional
    premises: list[str]  # Optional
    confidence: float    # Optional


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
    collection_name: str

    task_type: Literal["logic", "physics"]
    requested_task_type: NotRequired[Literal["logic", "physics"] | None]

    context: str
    intermediate_answer: IntermediateAnswer

    final_answer: FinalAnswer

    error: str

    retry_count: NotRequired[int]
