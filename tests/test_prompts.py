"""Test prompts — bao dam template format duoc, khong vo tinh xoa placeholder."""
from src.agent.prompts.logic_formalizer import Z3_USER_TEMPLATE
from src.agent.prompts.logic_explanation import (
    LOGIC_OUTPUT_PROMPT,
    LOGIC_OUTPUT_ERROR_PROMPT,
)
from src.agent.prompts.physics_formalizer import PHYSICS_USER_TEMPLATE
from src.agent.prompts.physics_explanation import (
    PHYSICS_OUTPUT_PROMPT,
    PHYSICS_OUTPUT_ERROR_PROMPT,
)


def test_logic_user_template():
    out = Z3_USER_TEMPLATE.format(premises_block="Premises:\n- A\n\n", question="Q?")
    assert "Premises:" in out and "Q?" in out


def test_logic_user_template_no_premises():
    out = Z3_USER_TEMPLATE.format(premises_block="", question="Q?")
    assert out.startswith("Logic Problem:")


def test_logic_output_prompt():
    out = LOGIC_OUTPUT_PROMPT.format(question="Q?", code_output="ANSWER: Yes")
    assert "ANSWER: Yes" in out


def test_logic_output_error_prompt():
    out = LOGIC_OUTPUT_ERROR_PROMPT.format(
        question="Q?",
        premises_block="- A",
        generated_code="bad code",
        error_message="SyntaxError",
    )
    assert "SyntaxError" in out and "bad code" in out


def test_physics_user_template():
    out = PHYSICS_USER_TEMPLATE.format(context_block="", question="V=?")
    assert "V=?" in out
    out2 = PHYSICS_USER_TEMPLATE.format(
        context_block="Relevant Formulas/Examples:\nF=ma\n\n",
        question="V=?",
    )
    assert "F=ma" in out2


def test_physics_output_prompts():
    s = PHYSICS_OUTPUT_PROMPT.format(question="Q", code_output="x=1")
    assert "x=1" in s

    e = PHYSICS_OUTPUT_ERROR_PROMPT.format(
        question="Q",
        context_block="",
        generated_code="bad",
        error_message="boom",
    )
    assert "boom" in e
