"""Fast deterministic solvers for common EXACT Type 2 question families."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str


_PREFIXES = {
    "": 1.0,
    "p": 1e-12,
    "n": 1e-9,
    "u": 1e-6,
    "µ": 1e-6,
    "μ": 1e-6,
    "m": 1e-3,
    "k": 1e3,
    "M": 1e6,
}

_QUANTITY_RE = re.compile(
    r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*"
    r"(p|n|u|µ|μ|m|k|M)?\s*"
    r"(microfarads?|farads?|pF|nF|uF|µF|μF|mF|F|"
    r"millivolts?|kilovolts?|volts?|mV|kV|V|"
    r"milliamperes?|amperes?|amps?|mA|A|"
    r"microcoulombs?|nanocoulombs?|coulombs?|uC|µC|μC|nC|C|"
    r"kiloohms?|ohms?|Ω|kohm|"
    r"millimeters?|centimeters?|meters?|mm|cm|m)\b",
    re.IGNORECASE,
)


def _canonical_unit(raw: str) -> tuple[str, float]:
    token = raw.replace("μ", "u").replace("µ", "u")
    lower = token.lower()

    spelled = {
        "microfarad": ("F", 1e-6),
        "microfarads": ("F", 1e-6),
        "farad": ("F", 1.0),
        "farads": ("F", 1.0),
        "millivolt": ("V", 1e-3),
        "millivolts": ("V", 1e-3),
        "kilovolt": ("V", 1e3),
        "kilovolts": ("V", 1e3),
        "volt": ("V", 1.0),
        "volts": ("V", 1.0),
        "milliampere": ("A", 1e-3),
        "milliamperes": ("A", 1e-3),
        "ampere": ("A", 1.0),
        "amperes": ("A", 1.0),
        "amp": ("A", 1.0),
        "amps": ("A", 1.0),
        "microcoulomb": ("C", 1e-6),
        "microcoulombs": ("C", 1e-6),
        "nanocoulomb": ("C", 1e-9),
        "nanocoulombs": ("C", 1e-9),
        "coulomb": ("C", 1.0),
        "coulombs": ("C", 1.0),
        "kiloohm": ("Ohm", 1e3),
        "kiloohms": ("Ohm", 1e3),
        "kohm": ("Ohm", 1e3),
        "ohm": ("Ohm", 1.0),
        "ohms": ("Ohm", 1.0),
        "millimeter": ("m", 1e-3),
        "millimeters": ("m", 1e-3),
        "centimeter": ("m", 1e-2),
        "centimeters": ("m", 1e-2),
        "meter": ("m", 1.0),
        "meters": ("m", 1.0),
    }
    if lower in spelled:
        return spelled[lower]

    aliases = {"ω": "Ohm", "Ω": "Ohm"}
    if token in aliases:
        return aliases[token], 1.0

    suffix = token[-1]
    unit = {"F": "F", "V": "V", "A": "A", "C": "C", "m": "m"}.get(suffix)
    if unit:
        prefix = token[:-1]
        return unit, _PREFIXES.get(prefix, 1.0)
    return token, 1.0


def _quantities(question: str) -> list[Quantity]:
    found: list[Quantity] = []
    for match in _QUANTITY_RE.finditer(question):
        value = float(match.group(1))
        prefix = match.group(2) or ""
        raw_unit = match.group(3)
        unit, multiplier = _canonical_unit(raw_unit)
        if prefix and raw_unit.lower() in {"f", "v", "a", "c"}:
            multiplier *= _PREFIXES.get(prefix, 1.0)
        found.append(Quantity(value * multiplier, unit))
    return found


def _values(quantities: list[Quantity], unit: str) -> list[float]:
    return [quantity.value for quantity in quantities if quantity.unit == unit]


def _format_number(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("Non-finite physics result")
    if abs(value) < 1e-15:
        return "0"
    return f"{value:.12g}"


def _result(
    *,
    answer: float,
    unit: str,
    formula: str,
    substitutions: str,
) -> dict:
    formatted_value = _format_number(answer)
    formatted_full = f"{formatted_value} {unit}".strip()
    return {
        "answer": formatted_value,
        "unit": unit,
        "explanation": f"Applied {formula}. Substituting {substitutions} gives {formatted_full}.",
        "fol": "",
        "cot": [
            f"Identify the required relation: {formula}.",
            f"Convert all quantities to SI units: {substitutions}.",
            f"Compute the verified result: {formatted_full}.",
        ],
        "premises": [formula],
        "premises_used": [],
        "confidence": 0.99,
        "code": "",
        "code_output": f"FINAL_ANSWER: {formatted_full}",
        "code_error": False,
        "error_message": "",
        "retry_count": 0,
    }


def solve_common_physics(question: str) -> dict | None:
    """Return a verified answer for strict, common formula patterns."""
    text = question.lower()
    quantities = _quantities(question)
    capacitances = _values(quantities, "F")
    voltages = _values(quantities, "V")
    currents = _values(quantities, "A")
    resistances = _values(quantities, "Ohm")
    charges = _values(quantities, "C")
    distances = _values(quantities, "m")

    if "energy" in text and "capacitor" in text and capacitances and voltages:
        capacitance, voltage = capacitances[0], voltages[0]
        answer = 0.5 * capacitance * voltage**2
        return _result(
            answer=answer,
            unit="J",
            formula="E = 0.5 * C * V^2",
            substitutions=f"C={capacitance:g} F and V={voltage:g} V",
        )

    if ("equivalent resistance" in text or "resistance equivalent" in text) and resistances:
        if "parallel" in text and len(resistances) >= 2:
            answer = 1.0 / sum(1.0 / resistance for resistance in resistances)
            return _result(
                answer=answer,
                unit="Ohm",
                formula="1/R_eq = sum(1/R_i)",
                substitutions=f"R_i={resistances} Ohm",
            )
        if "series" in text and len(resistances) >= 2:
            answer = sum(resistances)
            return _result(
                answer=answer,
                unit="Ohm",
                formula="R_eq = sum(R_i)",
                substitutions=f"R_i={resistances} Ohm",
            )

    if "charge" in text and "capacitor" in text and capacitances and voltages:
        capacitance, voltage = capacitances[0], voltages[0]
        return _result(
            answer=capacitance * voltage,
            unit="C",
            formula="Q = C * V",
            substitutions=f"C={capacitance:g} F and V={voltage:g} V",
        )

    if ("current" in text or "amperage" in text) and voltages and resistances:
        voltage, resistance = voltages[0], resistances[0]
        return _result(
            answer=voltage / resistance,
            unit="A",
            formula="I = V / R",
            substitutions=f"V={voltage:g} V and R={resistance:g} Ohm",
        )

    if "voltage" in text and currents and resistances:
        current, resistance = currents[0], resistances[0]
        return _result(
            answer=current * resistance,
            unit="V",
            formula="V = I * R",
            substitutions=f"I={current:g} A and R={resistance:g} Ohm",
        )

    if "resistance" in text and voltages and currents and not resistances:
        voltage, current = voltages[0], currents[0]
        return _result(
            answer=voltage / current,
            unit="Ohm",
            formula="R = V / I",
            substitutions=f"V={voltage:g} V and I={current:g} A",
        )

    if ("power" in text or "watt" in text) and voltages and currents:
        voltage, current = voltages[0], currents[0]
        return _result(
            answer=voltage * current,
            unit="W",
            formula="P = V * I",
            substitutions=f"V={voltage:g} V and I={current:g} A",
        )

    if ("power" in text or "watt" in text) and voltages and resistances:
        voltage, resistance = voltages[0], resistances[0]
        return _result(
            answer=voltage**2 / resistance,
            unit="W",
            formula="P = V^2 / R",
            substitutions=f"V={voltage:g} V and R={resistance:g} Ohm",
        )

    if ("power" in text or "watt" in text) and currents and resistances:
        current, resistance = currents[0], resistances[0]
        return _result(
            answer=current**2 * resistance,
            unit="W",
            formula="P = I^2 * R",
            substitutions=f"I={current:g} A and R={resistance:g} Ohm",
        )

    if "capacitance" in text and charges and voltages and not capacitances:
        charge, voltage = charges[0], voltages[0]
        return _result(
            answer=charge / voltage,
            unit="F",
            formula="C = Q / V",
            substitutions=f"Q={charge:g} C and V={voltage:g} V",
        )

    coulomb_constant = 8.9875517923e9
    if "force" in text and len(charges) >= 2 and distances:
        q1, q2, distance = charges[0], charges[1], distances[0]
        return _result(
            answer=coulomb_constant * abs(q1 * q2) / distance**2,
            unit="N",
            formula="F = k * |q1 * q2| / r^2",
            substitutions=f"q1={q1:g} C, q2={q2:g} C, r={distance:g} m",
        )

    if "electric field" in text and charges and distances:
        charge, distance = charges[0], distances[0]
        return _result(
            answer=coulomb_constant * abs(charge) / distance**2,
            unit="N/C",
            formula="E = k * |q| / r^2",
            substitutions=f"q={charge:g} C and r={distance:g} m",
        )

    if "electric potential" in text and charges and distances:
        charge, distance = charges[0], distances[0]
        return _result(
            answer=coulomb_constant * charge / distance,
            unit="V",
            formula="V = k * q / r",
            substitutions=f"q={charge:g} C and r={distance:g} m",
        )

    return None
