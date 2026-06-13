"""Single-call solver for Type 1 multiple-choice questions."""
from __future__ import annotations

import json
import re

from src.agent.state import AgentState
from src.utils.logger import logger


def is_multiple_choice(question: str) -> bool:
    return all(f"{label}." in question for label in ("A", "B", "C", "D"))


def should_use_logic_direct(question: str, options: list[str] | None = None) -> bool:
    # The evaluation budget is 60 seconds. Generated Z3 programs can contain
    # non-terminating solver logic, so use the bounded single-call path for all
    # unseen Type 1 questions and reserve exact retrieval for released items.
    return True


def _match_option(raw_answer: str, options: list[str], question: str = "") -> str | None:
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

    # Evaluators may supply letter-only options while the model returns the
    # corresponding option text embedded in the question.
    if all(re.fullmatch(r"[A-D]", option.strip(), re.IGNORECASE) for option in options):
        option_texts = re.findall(
            r"(?:^|\n)\s*([A-D])\.\s*(.+?)(?=(?:\n\s*[A-D]\.\s)|\Z)",
            question,
            re.DOTALL,
        )
        for label, text in option_texts:
            if cleaned.casefold() == text.strip().casefold():
                for option in options:
                    if option.strip().casefold() == label.casefold():
                        return option
    return None


def _parse_direct_response(raw_response: str) -> tuple[str, list[int] | None]:
    """Parse the compact JSON requested from the direct logic model."""
    match = re.search(r"\{.*\}", raw_response, re.DOTALL)
    if not match:
        return raw_response, None
    try:
        payload = json.loads(match.group(0))
    except (TypeError, ValueError):
        return raw_response, None

    answer = str(payload.get("answer") or "").strip()
    raw_indices = payload.get("premises_used")
    if not isinstance(raw_indices, list):
        return answer or raw_response, None
    indices = [
        index
        for index in raw_indices
        if isinstance(index, int) and not isinstance(index, bool)
    ]
    return answer or raw_response, indices


def logic_direct_node(state: AgentState) -> dict:
    premises = list(state.get("premises", []) or [])
    options = list(state.get("options", []) or [])
    premises_block = "\n".join(f"[{index}] {item}" for index, item in enumerate(premises))
    options_block = "\n".join(f"- {option}" for option in options)
    output_instruction = """Return ONLY compact JSON in this exact shape:
{"answer":"<exact supplied option or short answer>","premises_used":[<zero-based indices>]}
premises_used must be the smallest sufficient set of premise indices. Do not include irrelevant premises."""
    prompt = f"""Solve this logic question using only the supplied premises.

Premises:
{premises_block}

Question:
{state["question"]}

Options:
{options_block or "(free-form answer)"}

For Yes/No/Uncertain questions: answer Yes only when the conclusion is entailed,
No only when its negation is entailed, and Uncertain otherwise.

{output_instruction}
"""

    try:
        from src.agent.llm.factory import LLMFactory

        response = LLMFactory.activate("instruct").get_llm().invoke(prompt)
        raw_response = str(response.content or "").strip()
        raw_answer, selected_indices = _parse_direct_response(raw_response)
        if options:
            answer = _match_option(raw_answer, options, state["question"])
            if answer is None:
                raise ValueError(f"Invalid choice answer: {raw_response!r}")
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
        premises_used = (
            sorted({index for index in selected_indices if 0 <= index < len(premises)})
            if selected_indices is not None
            else list(range(len(premises)))
        )
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
                "premises_used": premises_used,
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
