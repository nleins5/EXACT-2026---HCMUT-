"""
Robust extractor for Python code embedded in LLM output.

Handles:
- Reasoning-model wrappers like ``<think>...</think>`` (DeepSeek-R1, o1-like).
- Closed fenced blocks ``` ```python ... ``` ```.
- Open / truncated fences (max_tokens cut mid-block) — extracts everything after
  the opening fence.
- Plain code without fences (heuristic last-resort).

Returns ``""`` when nothing that plausibly looks like Python can be recovered,
so that downstream solver nodes will set ``code_error=True`` instead of feeding
prose into ``subprocess.run`` (which then dies with a useless SyntaxError).
"""
from __future__ import annotations

import re

# Strip <think>...</think> blocks (greedy across newlines, case-insensitive).
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# A closed fenced code block. Language tag is optional.
_FENCE_CLOSED_RE = re.compile(
    r"```(?:python|py)?[ \t]*\n(.*?)\n?```",
    re.DOTALL | re.IGNORECASE,
)

# An OPEN (unterminated) fenced block — happens when LLM hits max_tokens.
_FENCE_OPEN_RE = re.compile(
    r"```(?:python|py)?[ \t]*\n(.+)$",
    re.DOTALL | re.IGNORECASE,
)

# Tokens that strongly suggest the text is actual Python source.
_CODE_HINT_TOKENS = (
    "import ", "from ",
    "def ", "class ",
    "Solver(", "ForAll(", "Implies(", "Bool(", "Int(", "Real(", "Const(",
    "sympy", "symbols(", "solve(",
    "print(",
)


def extract_python_code(text: str) -> str:
    """Best-effort extraction of executable Python from arbitrary LLM output.

    Args:
        text: Raw text returned by the model (may contain reasoning blocks,
            multiple fences, prose, or be truncated mid-block).

    Returns:
        A code string ready to feed into ``subprocess`` / ``exec``, or ``""``
        if no plausible code could be recovered.
    """
    if not text:
        return ""

    # 1. Strip reasoning blocks (DeepSeek-R1 etc. always emit them, regardless of prompt).
    cleaned = _THINK_RE.sub("", text).strip()

    # 2. Closed fenced block — preferred path.
    m = _FENCE_CLOSED_RE.search(cleaned)
    if m:
        return m.group(1).strip()

    # 3. Open fence (truncated by max_tokens). Take everything after the opener.
    m = _FENCE_OPEN_RE.search(cleaned)
    if m:
        return m.group(1).strip()

    # 4. No fence at all. Only treat as code if it actually looks like code,
    #    otherwise we'd be feeding prose to subprocess.
    if any(token in cleaned for token in _CODE_HINT_TOKENS):
        return cleaned

    return ""
