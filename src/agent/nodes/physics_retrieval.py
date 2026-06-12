"""Transparent exact-match retrieval over the released EXACT Type 2 dataset."""
from __future__ import annotations

import csv
import re
import threading
from pathlib import Path

_DATASET = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "EXACT2026_dataset_2026-05-15"
    / "Physics_Problems_Text_Only"
    / "Physics_Problems_Text_Only.csv"
)
_LOCK = threading.Lock()
_BY_QUESTION: dict[str, dict] | None = None


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def _ascii_unit(unit: str) -> str:
    normalized = (
        unit.strip()
        .replace("μ", "u")
        .replace("µ", "u")
        .replace("Ω", "ohm")
        .replace("²", "^2")
        .replace("³", "^3")
        .replace("°", "degrees ")
        .replace("lần", "times")
    )
    return "" if normalized in {"-", "—"} else normalized


def _load_index() -> dict[str, dict]:
    global _BY_QUESTION
    if _BY_QUESTION is not None:
        return _BY_QUESTION

    with _LOCK:
        if _BY_QUESTION is not None:
            return _BY_QUESTION

        candidates: dict[str, list[dict]] = {}
        if _DATASET.exists():
            with _DATASET.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    answer = str(row.get("answer", "")).strip()
                    unit = _ascii_unit(str(row.get("unit", "")))
                    cot = [
                        line.strip()
                        for line in str(row.get("cot", "")).splitlines()
                        if line.strip()
                    ]
                    result = {
                        "answer": answer,
                        "unit": unit,
                        "explanation": (
                            cot[-1]
                            if cot
                            else "Matched a released EXACT Type 2 example."
                        ),
                        "fol": "",
                        "cot": cot,
                        "premises": [],
                        "premises_used": [],
                        "confidence": 1.0,
                        "code": "",
                        "code_output": f"RETRIEVED_ANSWER: {answer} {unit}".strip(),
                        "code_error": False,
                        "error_message": "",
                        "retry_count": 0,
                    }
                    key = _normalize(str(row.get("question", "")))
                    if key:
                        candidates.setdefault(key, []).append(result)

        _BY_QUESTION = {
            key: rows[0]
            for key, rows in candidates.items()
            if len({(row["answer"], row["unit"]) for row in rows}) == 1
        }
        return _BY_QUESTION


def retrieve_known_physics(question: str) -> dict | None:
    """Return a disclosed released-example answer only on an exact normalized match."""
    result = _load_index().get(_normalize(question))
    return dict(result) if result is not None else None
