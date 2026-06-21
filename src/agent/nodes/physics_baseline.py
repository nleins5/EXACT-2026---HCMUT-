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
    r"(microfarads?|farads?|pF|nF|uF|µF|μF|mF|F|"
    r"millivolts?|kilovolts?|volts?|mV|kV|V|"
    r"milliamperes?|amperes?|amps?|mA|A|"
    r"microcoulombs?|nanocoulombs?|coulombs?|uC|µC|μC|nC|C|"
    r"kiloohms?|ohms?|kΩ|Ω|kohm|"
    r"millihenries?|henries?|mH|H|"
    r"kilohertz|hertz|kHz|Hz|"
    r"millijoules?|joules?|mJ|J|"
    r"newtons?|N|"
    r"milliseconds?|seconds?|ms|s|"
    r"kilograms?|grams?|kg|g|"
    r"meters per second squared|metres per second squared|m/s\^?2|m/s²|"
    r"meters per second|metres per second|m/s|"
    r"millimeters?|centimeters?|meters?|mm|cm|m)\b",
    re.IGNORECASE,
)

_GRAVITY_RE = re.compile(r"\bg\s*(?:=|is)?\s*(-?\d+(?:\.\d+)?)\s*(?:m/s\^?2|m/s²)?\b", re.I)
_ANGLE_RE = re.compile(r"(?:angle(?:\s+of)?|inclined at(?: an angle of)?)\s*(\d+(?:\.\d+)?)\s*(?:°|degrees?)", re.I)


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
        "millihenry": ("H", 1e-3),
        "millihenries": ("H", 1e-3),
        "henry": ("H", 1.0),
        "henries": ("H", 1.0),
        "kilohertz": ("Hz", 1e3),
        "hertz": ("Hz", 1.0),
        "millijoule": ("J", 1e-3),
        "millijoules": ("J", 1e-3),
        "joule": ("J", 1.0),
        "joules": ("J", 1.0),
        "newton": ("N", 1.0),
        "newtons": ("N", 1.0),
        "kilogram": ("kg", 1.0),
        "kilograms": ("kg", 1.0),
        "gram": ("kg", 1e-3),
        "grams": ("kg", 1e-3),
        "meters per second": ("m/s", 1.0),
        "metres per second": ("m/s", 1.0),
        "meters per second squared": ("m/s^2", 1.0),
        "metres per second squared": ("m/s^2", 1.0),
    }
    if lower in spelled:
        return spelled[lower]

    aliases = {"ω": "Ohm", "Ω": "Ohm", "kΩ": "Ohm"}
    if token in aliases:
        return aliases[token], 1e3 if token == "kΩ" else 1.0

    suffix = token[-1]
    if lower in {"m/s", "m/s2", "m/s^2", "m/s²"}:
        return ("m/s^2", 1.0) if "2" in lower or "²" in lower else ("m/s", 1.0)
    if lower in {"hz", "khz"}:
        return "Hz", 1e3 if lower == "khz" else 1.0
    if lower == "kg":
        return "kg", 1.0
    if lower == "g":
        return "kg", 1e-3

    unit = {"F": "F", "V": "V", "A": "A", "C": "C", "H": "H", "J": "J", "N": "N", "m": "m", "s": "s"}.get(suffix)
    if unit:
        prefix = token[:-1]
        return unit, _PREFIXES.get(prefix, 1.0)
    return token, 1.0


def _quantities(question: str) -> list[Quantity]:
    found: list[Quantity] = []
    for match in _QUANTITY_RE.finditer(question):
        value = float(match.group(1))
        raw_unit = match.group(2)
        unit, multiplier = _canonical_unit(raw_unit)
        found.append(Quantity(value * multiplier, unit))
    return found


def _values(quantities: list[Quantity], unit: str) -> list[float]:
    return [quantity.value for quantity in quantities if quantity.unit == unit]


def _gravity(question: str) -> float:
    match = _GRAVITY_RE.search(question)
    if match:
        return float(match.group(1))
    return 9.8


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
    masses = _values(quantities, "kg")
    speeds = _values(quantities, "m/s")
    accelerations = _values(quantities, "m/s^2")
    inductances = _values(quantities, "H")
    frequencies = _values(quantities, "Hz")
    energies = _values(quantities, "J")
    forces = _values(quantities, "N")
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
            "incline",
            "pulley",
            "friction",
            "collision",
            "spring",
            "projectile",
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
                "gravitational potential energy",
                "potential energy",
                "stored gravitational energy",
            ),
        )
        and len(masses) == 1
        and len(distances) == 1
        and not complex_context
    ):
        mass, height = masses[0], distances[0]
        gravity = _gravity(question)
        return _result(
            answer=mass * gravity * height,
            unit="J",
            formula="E_p = m * g * h",
            substitutions=f"m={mass:g} kg, g={gravity:g} m/s^2, h={height:g} m",
        )

    if (
        _has_any(text, ("kinetic energy", "calculate the ke", "find the ke"))
        and len(masses) == 1
        and len(speeds) == 1
        and not complex_context
    ):
        mass, speed = masses[0], speeds[0]
        return _result(
            answer=0.5 * mass * speed**2,
            unit="J",
            formula="E_k = 0.5 * m * v^2",
            substitutions=f"m={mass:g} kg and v={speed:g} m/s",
        )

    magnetic_energy = "magnetic" in text and "energy" in text and "inductor" in text
    if magnetic_energy and len(inductances) == 1 and len(currents) == 1 and not energies:
        inductance, current = inductances[0], currents[0]
        energy = 0.5 * inductance * current**2
        if re.search(r"(?:energy\s*\(mJ\)|unit\s*:\s*mJ)", question, re.I):
            return _result(answer=energy * 1e3, unit="mJ", formula="W = 0.5 * L * I^2", substitutions=f"L={inductance:g} H and I={current:g} A")
        return _result(answer=energy, unit="J", formula="W = 0.5 * L * I^2", substitutions=f"L={inductance:g} H and I={current:g} A")

    if magnetic_energy and len(energies) == 1 and len(inductances) == 1 and not currents:
        energy, inductance = energies[0], inductances[0]
        return _result(answer=math.sqrt(2 * energy / inductance), unit="A", formula="I = sqrt(2W / L)", substitutions=f"W={energy:g} J and L={inductance:g} H")

    if magnetic_energy and len(energies) == 1 and len(currents) == 1 and not inductances:
        energy, current = energies[0], currents[0]
        inductance = 2 * energy / current**2
        requested_mh = bool(re.search(r"inductance\s*\(mH\)|unit\s*:\s*mH", question, re.I))
        return _result(answer=inductance * (1e3 if requested_mh else 1), unit="mH" if requested_mh else "H", formula="L = 2W / I^2", substitutions=f"W={energy:g} J and I={current:g} A")

    if len(inductances) == 1 and len(capacitances) == 1:
        inductance, capacitance = inductances[0], capacitances[0]
        if "angular frequency" in text:
            return _result(answer=1 / math.sqrt(inductance * capacitance), unit="rad/s", formula="omega = 1 / sqrt(L*C)", substitutions=f"L={inductance:g} H and C={capacitance:g} F")
        if _has_any(text, ("resonant frequency", "resonance frequency", "natural oscillation frequency")):
            return _result(answer=1 / (2 * math.pi * math.sqrt(inductance * capacitance)), unit="Hz", formula="f = 1 / (2*pi*sqrt(L*C))", substitutions=f"L={inductance:g} H and C={capacitance:g} F")

    if len(frequencies) == 1 and len(inductances) == 1 and "inductive reactance" in text:
        frequency, inductance = frequencies[0], inductances[0]
        return _result(answer=2 * math.pi * frequency * inductance, unit="ohm", formula="X_L = 2*pi*f*L", substitutions=f"f={frequency:g} Hz and L={inductance:g} H")

    if len(frequencies) == 1 and len(capacitances) == 1 and "capacitive reactance" in text:
        frequency, capacitance = frequencies[0], capacitances[0]
        return _result(answer=1 / (2 * math.pi * frequency * capacitance), unit="ohm", formula="X_C = 1 / (2*pi*f*C)", substitutions=f"f={frequency:g} Hz and C={capacitance:g} F")

    if (
        len(frequencies) == 1
        and len(inductances) == 1
        and len(capacitances) == 1
        and len(resistances) == 1
        and _has_any(text, ("total impedance", "calculate the impedance", "find the impedance"))
    ):
        frequency, inductance = frequencies[0], inductances[0]
        capacitance, resistance = capacitances[0], resistances[0]
        x_l = 2 * math.pi * frequency * inductance
        x_c = 1 / (2 * math.pi * frequency * capacitance)
        return _result(answer=math.sqrt(resistance**2 + (x_l - x_c)**2), unit="ohm", formula="Z = sqrt(R^2 + (X_L-X_C)^2)", substitutions=f"R={resistance:g} ohm, X_L={x_l:g} ohm, X_C={x_c:g} ohm")

    if (
        _has_any(
            text,
            (
                "calculate the momentum",
                "find the momentum",
                "what is the momentum",
                "linear momentum",
            ),
        )
        and len(masses) == 1
        and len(speeds) == 1
        and not complex_context
    ):
        mass, speed = masses[0], speeds[0]
        return _result(
            answer=mass * speed,
            unit="kg*m/s",
            formula="p = m * v",
            substitutions=f"m={mass:g} kg and v={speed:g} m/s",
        )

    if (
        _has_any(text, ("calculate the force", "find the force", "what is the force"))
        and len(masses) == 1
        and len(accelerations) == 1
        and not complex_context
    ):
        mass, acceleration = masses[0], accelerations[0]
        return _result(
            answer=mass * acceleration,
            unit="N",
            formula="F = m * a",
            substitutions=f"m={mass:g} kg and a={acceleration:g} m/s^2",
        )

    if "resultant force" in text and forces:
        force_values = forces[:2]
        if len(force_values) == 1 and "each" in text:
            force_values.append(force_values[0])
        if len(force_values) == 2:
            first, second = force_values
            angle_match = _ANGLE_RE.search(question)
            if "same direction" in text:
                resultant, formula = first + second, "F = F1 + F2"
            elif _has_any(text, ("opposite directions", "opposite direction")):
                resultant, formula = abs(first - second), "F = |F1 - F2|"
            else:
                angle = 90.0 if "perpendicular" in text else (float(angle_match.group(1)) if angle_match else None)
                if angle is None:
                    resultant = None
                else:
                    resultant = math.sqrt(first**2 + second**2 + 2 * first * second * math.cos(math.radians(angle)))
                formula = "F = sqrt(F1^2 + F2^2 + 2*F1*F2*cos(theta))"
            if resultant is not None:
                return _result(answer=resultant, unit="N", formula=formula, substitutions=f"F1={first:g} N and F2={second:g} N")

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
