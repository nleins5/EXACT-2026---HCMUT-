"""Test LLMFactory + Supervisor swap mechanics — dung fake supervisor."""
from unittest.mock import MagicMock

import pytest

from src.agent.llm.factory import LLMFactory


@pytest.fixture(autouse=True)
def reset_factory():
    """Reset state truoc moi test (singleton)."""
    LLMFactory.reset()
    yield
    LLMFactory.reset()


def test_activate_without_init_raises():
    LLMFactory.reset()
    with pytest.raises(RuntimeError, match="chua duoc init"):
        LLMFactory.activate("coder")


def test_activate_calls_swap_to_first_time():
    fake_sup = MagicMock()
    LLMFactory.init(fake_sup)

    client = LLMFactory.activate("coder")
    assert client is not None
    assert client.role == "coder"
    fake_sup.swap_to.assert_called_once_with("coder")


def test_activate_same_role_no_swap():
    fake_sup = MagicMock()
    LLMFactory.init(fake_sup)

    LLMFactory.activate("coder")
    LLMFactory.activate("coder")  # same role
    LLMFactory.activate("coder")

    # Chi nen swap 1 lan total.
    assert fake_sup.swap_to.call_count == 1


def test_activate_role_change_triggers_swap():
    fake_sup = MagicMock()
    LLMFactory.init(fake_sup)

    LLMFactory.activate("coder")
    LLMFactory.activate("instruct")
    LLMFactory.activate("coder")

    # 3 lan goi voi 3 role thay phien -> 3 swap.
    assert fake_sup.swap_to.call_count == 3
    calls = [c.args[0] for c in fake_sup.swap_to.call_args_list]
    assert calls == ["coder", "instruct", "coder"]


def test_activate_returns_correct_role_client():
    fake_sup = MagicMock()
    LLMFactory.init(fake_sup)

    c1 = LLMFactory.activate("coder")
    c2 = LLMFactory.activate("instruct")
    assert c1.role == "coder"
    assert c2.role == "instruct"
    # Khac instance vi rebuild moi role.
    assert c1 is not c2
