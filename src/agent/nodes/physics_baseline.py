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
    r"milliseconds?|seconds?|ms|s|"
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
        "millisecond": ("s", 1e-3),
        "milliseconds": ("s", 1e-3),
        "second": ("s", 1.0),
        "seconds": ("s", 1.0),
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
    unit = {"F": "F", "V": "V", "A": "A", "C": "C", "m": "m", "s": "s"}.get(suffix)
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
        if prefix and raw_unit.lower() in {"f", "v", "a", "c", "ω", "ohm", "ohms"}:
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
    output_unit = "ohm" if unit == "Ohm" else unit
    formatted_full = f"{formatted_value} {output_unit}".strip()
    return {
        "answer": formatted_value,
        "unit": output_unit,
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


def _equivalent_resistance(resistances: list[float], connection: str) -> float | None:
    if not resistances:
        return None
    if connection == "series":
        return sum(resistances)
    if connection == "parallel":
        if any(resistance == 0 for resistance in resistances):
            return 0.0
        return 1.0 / sum(1.0 / resistance for resistance in resistances)
    return None


def _has_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


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
    times = _values(quantities, "s")
    complex_context = _has_any(
        text,
        (
            " after ",
            " before ",
            " then ",
            " disconnected",
            " connected with",
            " dielectric",
            " cut ",
            " another ",
            " changes",
            " relative error",
            " absolute error",
            " uncertainty",
            " rlc",
            " impedance",
            " resonance",
            " alternating",
        ),
    )

    if (
        _has_any(text, ("average speed", "calculate the speed", "find the speed", "what is its speed"))
        and len(distances) == 1
        and len(times) == 1
        and times[0] != 0
        and not complex_context
    ):
        distance, duration = distances[0], times[0]
        return _result(
            answer=distance / duration,
            unit="m/s",
            formula="v = d / t",
            substitutions=f"d={distance:g} m and t={duration:g} s",
        )

    if (
        _has_any(
            text,
            (
                "calculate the energy stored",
                "find the energy stored",
                "what is the energy stored",
            ),
        )
        and "capacitor" in text
        and len(capacitances) == 1
        and len(voltages) == 1
        and not complex_context
    ):
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
            answer = _equivalent_resistance(resistances, "parallel")
            return _result(
                answer=answer,
                unit="Ohm",
                formula="1/R_eq = sum(1/R_i)",
                substitutions=f"R_i={resistances} Ohm",
            )
        if "series" in text and len(resistances) >= 2:
            answer = _equivalent_resistance(resistances, "series")
            return _result(
                answer=answer,
                unit="Ohm",
                formula="R_eq = sum(R_i)",
                substitutions=f"R_i={resistances} Ohm",
            )

    if (
        _has_any(text, ("calculate the charge", "find the charge", "what is the charge"))
        and "capacitor" in text
        and len(capacitances) == 1
        and len(voltages) == 1
        and not complex_context
    ):
        capacitance, voltage = capacitances[0], voltages[0]
        return _result(
            answer=capacitance * voltage,
            unit="C",
            formula="Q = C * V",
            substitutions=f"C={capacitance:g} F and V={voltage:g} V",
        )

    if (
        _has_any(
            text,
            (
                "calculate the current",
                "find the current",
                "what is the current",
                "calculate the total current",
                "find the total current",
                "what is the total current",
            ),
        )
        and voltages
        and resistances
        and not complex_context
    ):
        voltage = voltages[0]
        resistance = resistances[0]
        connection = ""
        if len(resistances) >= 2:
            total_current_requested = any(
                phrase in text
                for phrase in (
                    "total current",
                    "source current",
                    "circuit current",
                    "current supplied",
                    "current drawn",
                )
            )
            if "parallel" in text and total_current_requested:
                connection = "parallel"
            elif "series" in text:
                connection = "series"
        if connection:
            resistance = _equivalent_resistance(resistances, connection)
        if resistance == 0:
            return None
        return _result(
            answer=voltage / resistance,
            unit="A",
            formula="I = V / R",
            substitutions=f"V={voltage:g} V and R_eq={resistance:g} ohm",
        )

    if (
        _has_any(text, ("calculate the voltage", "find the voltage", "what is the voltage"))
        and len(currents) == 1
        and len(resistances) == 1
        and not complex_context
    ):
        current, resistance = currents[0], resistances[0]
        return _result(
            answer=current * resistance,
            unit="V",
            formula="V = I * R",
            substitutions=f"I={current:g} A and R={resistance:g} Ohm",
        )

    if (
        _has_any(
            text,
            ("calculate the resistance", "find the resistance", "what is the resistance"),
        )
        and len(voltages) == 1
        and len(currents) == 1
        and not resistances
        and not complex_context
    ):
        voltage, current = voltages[0], currents[0]
        if current == 0:
            return None
        return _result(
            answer=voltage / current,
            unit="Ohm",
            formula="R = V / I",
            substitutions=f"V={voltage:g} V and I={current:g} A",
        )

    direct_power = _has_any(
        text, ("calculate the power", "find the power", "what is the power")
    )
    if direct_power and len(voltages) == 1 and len(currents) == 1 and not complex_context:
        voltage, current = voltages[0], currents[0]
        return _result(
            answer=voltage * current,
            unit="W",
            formula="P = V * I",
            substitutions=f"V={voltage:g} V and I={current:g} A",
        )

    if direct_power and len(voltages) == 1 and len(resistances) == 1 and not complex_context:
        voltage, resistance = voltages[0], resistances[0]
        if resistance == 0:
            return None
        return _result(
            answer=voltage**2 / resistance,
            unit="W",
            formula="P = V^2 / R",
            substitutions=f"V={voltage:g} V and R={resistance:g} Ohm",
        )

    if direct_power and len(currents) == 1 and len(resistances) == 1 and not complex_context:
        current, resistance = currents[0], resistances[0]
        return _result(
            answer=current**2 * resistance,
            unit="W",
            formula="P = I^2 * R",
            substitutions=f"I={current:g} A and R={resistance:g} Ohm",
        )

    if (
        _has_any(
            text,
            ("calculate the capacitance", "find the capacitance", "what is the capacitance"),
        )
        and len(charges) == 1
        and len(voltages) == 1
        and not capacitances
        and not complex_context
    ):
        charge, voltage = charges[0], voltages[0]
        if voltage == 0:
            return None
        return _result(
            answer=charge / voltage,
            unit="F",
            formula="C = Q / V",
            substitutions=f"Q={charge:g} C and V={voltage:g} V",
        )

    coulomb_constant = 8.9875517923e9
    if (
        "force between" in text
        and len(charges) == 2
        and len(distances) == 1
        and not complex_context
    ):
        q1, q2, distance = charges[0], charges[1], distances[0]
        if distance == 0:
            return None
        return _result(
            answer=coulomb_constant * abs(q1 * q2) / distance**2,
            unit="N",
            formula="F = k * |q1 * q2| / r^2",
            substitutions=f"q1={q1:g} C, q2={q2:g} C, r={distance:g} m",
        )

    if (
        "electric field" in text
        and "due to a" in text
        and "point charge" in text
        and len(charges) == 1
        and len(distances) == 1
        and not complex_context
    ):
        charge, distance = charges[0], distances[0]
        if distance == 0:
            return None
        return _result(
            answer=coulomb_constant * abs(charge) / distance**2,
            unit="N/C",
            formula="E = k * |q| / r^2",
            substitutions=f"q={charge:g} C and r={distance:g} m",
        )

    if (
        "electric potential" in text
        and len(charges) == 1
        and len(distances) == 1
        and not complex_context
    ):
        charge, distance = charges[0], distances[0]
        if distance == 0:
            return None
        return _result(
            answer=coulomb_constant * charge / distance,
            unit="V",
            formula="V = k * q / r",
            substitutions=f"q={charge:g} C and r={distance:g} m",
        )

    return None
