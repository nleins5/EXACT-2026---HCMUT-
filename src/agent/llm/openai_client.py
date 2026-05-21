"""OpenAI-compatible client tro ve `llama-server` local (BTC Q5).

Vi sao khong reuse llama-cpp-python truc tiep:
- BTC Q5 yeu cau /v1/models endpoint -> phai HTTP, khong duoc in-process.
- llama-server chuan OpenAI -> ChatOpenAI plug-and-play.

Port co dinh -> 1 instance ChatOpenAI dung cho moi role; chi can doi `model` header.
"""
from __future__ import annotations

from typing import Any, Type

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.agent.llm.base import BaseLLM
from src.core.config import settings
from src.utils.logger import logger


class OpenAILLMClient(BaseLLM):
    """Wrap `ChatOpenAI` voi cau hinh tro ve llama-server local.

    Args:
        role: "coder" hoac "instruct". Chon model_name + temperature + max_tokens
              tuong ung tu settings.llm.<role>.

    Note:
        Field `model` trong request HTTP se la `model_name` cua role.
        llama-server dung `--alias` de match field nay khi server start.
    """

    def __init__(self, role: str = "instruct"):
        cfg_server = settings.llm.server
        cfg_role = (
            settings.llm.coder if role == "coder"
            else settings.llm.instruct
        )

        self.role = role
        self.model_name = cfg_role.model_name
        self.temperature = cfg_role.temperature
        self.max_tokens = cfg_role.max_tokens
        self.base_url = cfg_server.base_url
        self.api_key = cfg_server.api_key

        self._llm: ChatOpenAI | None = None

        logger.debug(
            f"OpenAILLMClient init (role={role}, model={self.model_name}, "
            f"base_url={self.base_url})"
        )

    def get_llm(self, **kwargs) -> ChatOpenAI:
        """Tra ve `ChatOpenAI` singleton bound vao role hien tai.

        Args:
            **kwargs: forward toi `ChatOpenAI(...)` trong lan dau khoi tao.
                      Cac lan goi sau dung lai cache, kwargs bi ignore.
        """
        if self._llm is not None:
            return self._llm

        self._llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            base_url=self.base_url,
            api_key=self.api_key,
            **kwargs,
        )
        return self._llm

    def get_structured_llm(self, output_schema: Type[BaseModel]) -> Any:
        """Tra ve LLM voi structured output (Pydantic schema enforcement).

        Dung with_structured_output cua LangChain — tu dong ep kieu
        output theo Pydantic schema.
        """
        llm = self.get_llm()
        return llm.with_structured_output(output_schema)
