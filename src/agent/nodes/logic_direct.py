"""Single-call solver for Type 1 multiple-choice questions."""
from __future__ import annotations

import re

from src.agent.state import AgentState
from src.utils.logger import logger


def is_multiple_choice(question: str) -> bool:
    return all(f"{label}." in question for label in ("A", "B", "C", "D"))


def should_use_logic_direct(question: str, options: list[str] | None = None) -> bool:
    if is_multiple_choice(question):
        return True
    normalized_options = {option.strip().casefold() for option in options or []}
    truth_options = {"yes", "no", "unknown", "uncertain"}
    if normalized_options:
        return not normalized_options.issubset(truth_options)
    return bool(
        re.match(
            r"^\s*(?:how many|how much|what number|which|who|what is|what are)\b",
            question,
            re.IGNORECASE,
        )
    )


def _match_option(raw_answer: str, options: list[str]) -> str | None:
    cleaned = re.sub(
        r"^\s*(?:answer|option|choice)\s*(?:is|:|-)?\s*",
        "",
        raw_answer,
        flags=re.IGNORECASE,
    ).strip(" \t\r\n.\"'")
    for option in options:
        if cleaned.casefold() == option.strip().casefold():
            return option
    marker = re.search(r"\b(?:answer|option|choice)\s*(?:is|:|-)?\s*([A-D])\b", raw_answer, re.I)
    if marker:
        for option in options:
            if option.strip().casefold() == marker.group(1).casefold():
                return option
    return None


def logic_direct_node(state: AgentState) -> dict:
    premises = list(state.get("premises", []) or [])
    options = list(state.get("options", []) or [])
    premises_block = "\n".join(f"{index + 1}. {item}" for index, item in enumerate(premises))
    options_block = "\n".join(f"- {option}" for option in options)
    output_instruction = (
        "Return ONLY the exact text of one supplied option. No explanation or punctuation."
        if options
        else "Return ONLY the short final answer. No explanation, label, or punctuation."
    )
    prompt = f"""Solve this logic question using only the supplied premises.

Premises:
{premises_block}

Question:
{state["question"]}

Options:
{options_block or "(free-form answer)"}

{output_instruction}
"""

    try:
        from src.agent.llm.factory import LLMFactory

        response = LLMFactory.activate("instruct").get_llm().invoke(prompt)
        raw_answer = str(response.content or "").strip()
        if options:
            answer = _match_option(raw_answer, options)
            if answer is None:
                raise ValueError(f"Invalid choice answer: {raw_answer!r}")
        elif is_multiple_choice(state["question"]):
            match = re.search(r"\b(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:|-)?\s*([A-D])\b", raw_answer, re.I)
            if match is None and re.fullmatch(r"\s*[A-D]\s*", raw_answer, re.I):
                match = re.match(r"\s*([A-D])", raw_answer, re.I)
            if match is None:
                raise ValueError(f"Invalid multiple-choice answer: {raw_answer!r}")
            answer = match.group(1).upper()
        else:
            answer = re.sub(
                r"^\s*(?:answer|final answer)\s*(?:is|:|-)?\s*",
                "",
                raw_answer.splitlines()[0] if raw_answer else "",
                flags=re.IGNORECASE,
            ).strip(" \t\r\n.\"'")
            if not answer:
                raise ValueError("Empty direct answer")
        option_match = re.search(
            rf"^{answer}\.\s*(.+?)(?=^[A-D]\.\s|\Z)",
            state["question"],
            re.MULTILINE | re.DOTALL,
        )
        option_text = option_match.group(1).strip() if option_match else answer
        return {
            "final_answer": {
                "answer": answer,
                "explanation": (
                    f"The answer {answer} was selected after comparing the query "
                    f"against the supplied premises."
                ),
                "fol": "",
                "cot": [
                    "Compared each option against the supplied premises.",
                    f"Selected answer: {option_text}",
                ],
                "premises": premises,
                "premises_used": list(range(len(premises))),
                "unit": "",
                "confidence": 0.75,
            }
        }
    except Exception as exc:
        logger.error("logic_direct_node failed: %s", exc)
        return {
            "final_answer": {
                "answer": "Unknown",
                "explanation": "No option could be verified from the supplied premises.",
                "fol": "",
                "cot": ["Compared the answer options against the supplied premises."],
                "premises": premises,
                "premises_used": list(range(len(premises))),
                "unit": "",
                "confidence": 0.0,
            },
            "error": str(exc),
        }
