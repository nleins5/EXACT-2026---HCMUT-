"""Abstract base cho moi LLM client (chat + structured output)."""
from abc import ABC, abstractmethod
from typing import Any, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class BaseLLM(ABC):
    """Interface chung — cho phep node code khong phu thuoc provider cu the.

    Moi subclass phai expose:
    - get_llm(): tra ve LangChain chat model (cho free-form invoke).
    - get_structured_llm(schema): tra ve model bound vao Pydantic schema.
    """

    @abstractmethod
    def get_llm(self, **kwargs) -> Any:
        """Tra ve chat LLM instance.

        Args:
            **kwargs: forward toi provider client lan dau khoi tao.

        Returns:
            Chat model object (vi du ChatOpenAI) san sang `.invoke(messages)`.
        """

    @abstractmethod
    def get_structured_llm(self, output_schema: Type[T]) -> Any:
        """Tra ve LLM da bind vao schema (parse output thanh Pydantic obj).

        Args:
            output_schema: BaseModel subclass dinh nghia output mong doi.

        Returns:
            Model instance ma `.invoke(prompt)` tra ve instance cua `output_schema`.
        """
