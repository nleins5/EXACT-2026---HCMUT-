"""LLMFactory — single-resident gateway tro vao llama-server.

Hop dong su dung:
    # 1 lan tai startup (lifespan):
    supervisor = LlamaServerSupervisor()
    LLMFactory.init(supervisor)

    # Trong moi node:
    llm_client = LLMFactory.activate("coder")     # swap process neu can
    llm = llm_client.get_llm()
    response = llm.invoke([SystemMessage(...), HumanMessage(...)])

BTC Q3: tai 1 thoi diem chi 1 model resident -> activate() goi swap_to() noi bo.
"""
from __future__ import annotations

from typing import Literal, Optional

from src.agent.llm.openai_client import OpenAILLMClient
from src.agent.llm.server_supervisor import LlamaServerSupervisor
from src.utils.logger import logger

Role = Literal["coder", "instruct"]


class LLMFactory:
    """Singleton gateway. Phai goi `init(supervisor)` truoc khi `activate()`."""

    _supervisor: Optional[LlamaServerSupervisor] = None
    _client: Optional[OpenAILLMClient] = None
    _active_role: Optional[Role] = None

    @classmethod
    def init(cls, supervisor: LlamaServerSupervisor) -> None:
        """Gan supervisor (goi 1 lan o lifespan startup)."""
        cls._supervisor = supervisor
        cls._client = None
        cls._active_role = None
        logger.info("LLMFactory initialised voi LlamaServerSupervisor.")

    @classmethod
    def activate(cls, role: Role) -> OpenAILLMClient:
        """Dam bao llama-server chay role yeu cau, tra ve client tuong ung.

        Logic:
        - Neu role chua active -> supervisor.swap_to() (kill cu + spawn moi).
        - Rebuild OpenAILLMClient de model_name khop role moi (header `model`).
        - Neu role da active -> reuse client cache.

        Raises:
            RuntimeError: chua goi `init()`.
            FileNotFoundError / TimeoutError: tu supervisor.swap_to().
        """
        if cls._supervisor is None:
            raise RuntimeError(
                "LLMFactory chua duoc init. "
                "Goi LLMFactory.init(supervisor) trong app lifespan."
            )

        if cls._active_role != role:
            cls._supervisor.swap_to(role)
            cls._active_role = role
            cls._client = OpenAILLMClient(role=role)
            logger.info(
                f"LLMFactory.activate({role}) — process swapped, client rebuilt."
            )
        elif cls._client is None:
            cls._client = OpenAILLMClient(role=role)

        return cls._client

    @classmethod
    def reset(cls) -> None:
        """Test hook — clear toan bo state (KHONG kill supervisor process)."""
        cls._supervisor = None
        cls._client = None
        cls._active_role = None
