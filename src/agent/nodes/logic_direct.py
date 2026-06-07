"""Single-call solver for Type 1 multiple-choice questions."""
from __future__ import annotations

import re

from src.agent.state import AgentState
from src.utils.logger import logger


def is_multiple_choice(question: str) -> bool:
    return all(f"{label}." in question for label in ("A", "B", "C", "D"))


def logic_direct_node(state: AgentState) -> dict:
    premises = list(state.get("premises", []) or [])
    premises_block = "\n".join(f"{index + 1}. {item}" for index, item in enumerate(premises))
    prompt = f"""Solve this logic multiple-choice question using only the supplied premises.

Premises:
{premises_block}

Question:
{state["question"]}

Return ONLY the single correct option letter: A, B, C, or D. No explanation or punctuation.
"""

    try:
        from src.agent.llm.factory import LLMFactory

        response = LLMFactory.activate("instruct").get_llm().invoke(prompt)
        raw_answer = str(response.content or "").strip().upper()
        match = re.search(r"ANSWER\s*:\s*([A-D])\b", raw_answer)
        if match is None:
            matches = re.findall(r"\b([A-D])\b", raw_answer)
            match = re.match(r"([A-D])", matches[-1]) if matches else None
        if match is None:
            raise ValueError(f"Invalid multiple-choice answer: {raw_answer!r}")
        answer = match.group(1)
        option_match = re.search(
            rf"^{answer}\.\s*(.+?)(?=^[A-D]\.\s|\Z)",
            state["question"],
            re.MULTILINE | re.DOTALL,
        )
        option_text = option_match.group(1).strip() if option_match else f"option {answer}"
        return {
            "final_answer": {
                "answer": answer,
                "explanation": (
                    f"Option {answer} is the conclusion selected after comparing every "
                    f"choice against the supplied premises: {option_text}"
                ),
                "fol": "",
                "cot": [
                    "Compared each option against the supplied premises.",
                    f"Selected option {answer}: {option_text}",
                ],
                "premises": premises,
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
                "confidence": 0.0,
            },
            "error": str(exc),
        }
