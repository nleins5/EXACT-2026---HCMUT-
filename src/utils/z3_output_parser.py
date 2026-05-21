"""Parse Z3 solver output into a normalized prediction.

The fine-tuned coder model may output Z3 results in various formats depending
on training data version. This parser handles all known formats and normalizes
to a single prediction: "True", "False", or "Unknown".

Known output formats:
1. "Predicted: True/False/Unknown"                    (target format)
2. "Expected: True, Predicted: True"                  (old training format)
3. "Expected: True"                                   (model shortcut)
4. "Match: Conclusion 1 is entailed\\n..."            (multi-conclusion)
5. "Conclusion 1: [True]\\nConclusion 2: [False]..."  (another variant)
"""
import re
from typing import Literal

Prediction = Literal["True", "False", "Unknown"]


def parse_z3_output(raw_output: str) -> Prediction:
    """Parse Z3 solver stdout into normalized prediction.

    Priority order:
    1. Look for "Predicted: <X>" (canonical format).
    2. Look for "Expected: <X>, Predicted: <Y>" -> use Predicted.
    3. Look for "Expected: <X>" alone -> use as prediction (model shortcut).
    4. Look for multi-conclusion format -> majority vote.
    5. Fallback: "Unknown".

    Args:
        raw_output: stdout from Z3 subprocess execution.

    Returns:
        "True", "False", or "Unknown".
    """
    if not raw_output or not raw_output.strip():
        return "Unknown"

    text = raw_output.strip()

    # 1. Canonical: "Predicted: True/False/Unknown"
    m = re.search(r"Predicted:\s*(True|False|Unknown)", text, re.IGNORECASE)
    if m:
        return _normalize(m.group(1))

    # 2. Old format: "Expected: X, Predicted: Y" -> use Predicted
    m = re.search(r"Expected:\s*\w+,?\s*Predicted:\s*(True|False|Unknown)", text, re.IGNORECASE)
    if m:
        return _normalize(m.group(1))

    # 3. "Expected: True/False/Unknown" alone (model shortcut - uses Expected as output)
    m = re.search(r"Expected:\s*(True|False|Unknown)", text, re.IGNORECASE)
    if m:
        return _normalize(m.group(1))

    # 4. Multi-conclusion: "Conclusion N: [Entailed/NotEntailed/True/False]"
    conclusions = re.findall(
        r"Conclusion\s*\d+[:\s]+\[?(Entailed|NotEntailed|True|False|Unknown)\]?",
        text, re.IGNORECASE,
    )
    if conclusions:
        return _majority_vote(conclusions)

    # 5. "Match: Conclusion N is entailed/not entailed"
    matches = re.findall(
        r"Conclusion\s*\d+\s+is\s+(entailed|not entailed)",
        text, re.IGNORECASE,
    )
    if matches:
        return _majority_vote_entailment(matches)

    # 6. Simple True/False/Unknown on its own line
    m = re.search(r"^(True|False|Unknown)\s*$", text, re.MULTILINE | re.IGNORECASE)
    if m:
        return _normalize(m.group(1))

    # Fallback
    return "Unknown"


def _normalize(val: str) -> Prediction:
    """Normalize a prediction string to canonical form."""
    v = val.strip().lower()
    if v == "true":
        return "True"
    elif v == "false":
        return "False"
    return "Unknown"


def _majority_vote(conclusions: list[str]) -> Prediction:
    """Majority vote over multi-conclusion results.

    Entailed/True -> True; NotEntailed/False -> counts toward Unknown;
    If ALL are entailed -> True; if ALL are not-entailed -> check contradiction.
    Mixed -> Unknown.
    """
    entailed = 0
    not_entailed = 0
    for c in conclusions:
        c_lower = c.lower()
        if c_lower in ("entailed", "true"):
            entailed += 1
        elif c_lower in ("notentailed", "false", "unknown"):
            not_entailed += 1

    total = len(conclusions)
    if entailed == total:
        return "True"
    elif not_entailed == total:
        # All not-entailed could mean False or Unknown depending on context.
        # Conservative: return "Unknown" since not-entailed != contradicted.
        return "Unknown"
    else:
        return "Unknown"


def _majority_vote_entailment(matches: list[str]) -> Prediction:
    """Vote over 'is entailed' / 'is not entailed' matches."""
    entailed = sum(1 for m in matches if "not" not in m.lower())
    total = len(matches)
    if entailed == total:
        return "True"
    elif entailed == 0:
        return "Unknown"  # not entailed != contradicted
    return "Unknown"
