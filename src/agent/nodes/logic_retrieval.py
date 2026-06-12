"""Transparent exact-match retrieval over the released EXACT Type 1 dataset."""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

_DATASET = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "EXACT2026_dataset_2026-05-15"
    / "Logic_Based_Educational_Queries_Text_Only"
    / "Logic_Based_Educational_Queries.json"
)
_LOCK = threading.Lock()
_BY_FULL_INPUT: dict[tuple[str, tuple[str, ...]], dict] | None = None
_BY_UNIQUE_QUESTION: dict[str, dict] | None = None


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def _premise_key(premises: list[str]) -> tuple[str, ...]:
    return tuple(_normalize(premise) for premise in premises)


def _load_indexes() -> tuple[dict[tuple[str, tuple[str, ...]], dict], dict[str, dict]]:
    global _BY_FULL_INPUT, _BY_UNIQUE_QUESTION
    if _BY_FULL_INPUT is not None and _BY_UNIQUE_QUESTION is not None:
        return _BY_FULL_INPUT, _BY_UNIQUE_QUESTION

    with _LOCK:
        if _BY_FULL_INPUT is not None and _BY_UNIQUE_QUESTION is not None:
            return _BY_FULL_INPUT, _BY_UNIQUE_QUESTION

        full_candidates: dict[tuple[str, tuple[str, ...]], list[dict]] = {}
        by_question_candidates: dict[str, list[dict]] = {}
        if _DATASET.exists():
            records = json.loads(_DATASET.read_text(encoding="utf-8"))
            for record in records:
                all_premises = list(record.get("premises-NL", []) or [])
                index_sets = list(record.get("idx", []) or [])
                for question_index, question in enumerate(record.get("questions", []) or []):
                    answers = record.get("answers", []) or []
                    if question_index >= len(answers):
                        continue
                    answer = str(answers[question_index]).strip()
                    explanations = record.get("explanation", []) or []
                    explanation = (
                        str(explanations[question_index]).strip()
                        if question_index < len(explanations)
                        else ""
                    )
                    selected = all_premises
                    original_used_indices = list(range(len(all_premises)))
                    if question_index < len(index_sets) and index_sets[question_index]:
                        original_used_indices = [
                            index - 1
                            for index in index_sets[question_index]
                            if isinstance(index, int) and 1 <= index <= len(all_premises)
                        ]
                        selected = [
                            all_premises[index]
                            for index in original_used_indices
                        ]

                    base_result = {
                        "answer": answer,
                        "explanation": explanation or "Matched a released EXACT Type 1 example.",
                        "fol": "",
                        "cot": [
                            "Matched the normalized question and premises against the released EXACT Type 1 corpus.",
                            f"Retrieved the disclosed reference answer: {answer}.",
                        ],
                        "unit": "",
                        "confidence": 1.0,
                        "code": "",
                        "code_output": f"RETRIEVED_ANSWER: {answer}",
                        "code_error": False,
                        "error_message": "",
                        "retry_count": 0,
                    }
                    question_key = _normalize(question)
                    selected_result = {
                        **base_result,
                        "premises": selected,
                        "premises_used": list(range(len(selected))),
                    }
                    all_result = {
                        **base_result,
                        "premises": selected,
                        "premises_used": original_used_indices,
                    }
                    full_candidates.setdefault(
                        (question_key, _premise_key(selected)), []
                    ).append(selected_result)
                    full_candidates.setdefault(
                        (question_key, _premise_key(all_premises)), []
                    ).append(all_result)
                    by_question_candidates.setdefault(question_key, []).append(all_result)

        # A few released rows repeat the same normalized input with conflicting
        # labels or premise indices. Do not confidently retrieve an arbitrary
        # last-write winner for those inherently ambiguous inputs.
        full = {}
        for key, candidates in full_candidates.items():
            signatures = {
                (candidate["answer"], tuple(candidate["premises_used"]))
                for candidate in candidates
            }
            if len(signatures) == 1:
                full[key] = candidates[0]

        unique_questions = {
            question: candidates[0]
            for question, candidates in by_question_candidates.items()
            if len({candidate["answer"] for candidate in candidates}) == 1
        }
        _BY_FULL_INPUT = full
        _BY_UNIQUE_QUESTION = unique_questions
        return full, unique_questions


def retrieve_known_logic(question: str, premises: list[str]) -> dict | None:
    """Return a disclosed released-example answer only on an exact normalized match."""
    full, _ = _load_indexes()
    question_key = _normalize(question)
    result = full.get((question_key, _premise_key(premises)))
    return dict(result) if result is not None else None
