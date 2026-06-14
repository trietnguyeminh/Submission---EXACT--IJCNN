from __future__ import annotations

import json
import re
import sys
from typing import Any


SUPERSCRIPT_TRANSLATION = str.maketrans(
    {
        "\u2070": "0",
        "\u00b9": "1",
        "\u00b2": "2",
        "\u00b3": "3",
        "\u2074": "4",
        "\u2075": "5",
        "\u2076": "6",
        "\u2077": "7",
        "\u2078": "8",
        "\u2079": "9",
        "\u207b": "-",
        "\u2080": "0",
        "\u2081": "1",
        "\u2082": "2",
        "\u2083": "3",
        "\u2084": "4",
        "\u2085": "5",
        "\u2086": "6",
        "\u2087": "7",
        "\u2088": "8",
        "\u2089": "9",
    }
)

SUPERSCRIPT_EXPONENTS = {
    "\u2070": "0",
    "\u00b9": "1",
    "\u00b2": "2",
    "\u00b3": "3",
    "\u2074": "4",
    "\u2075": "5",
    "\u2076": "6",
    "\u2077": "7",
    "\u2078": "8",
    "\u2079": "9",
    "\u207b": "-",
    "\u207a": "+",
}

NUMBER_RE = (
    r"[-+]?(?:10\s*\^\s*\{?[-+]?\d+\}?"
    r"|\d+(?:\.\d+)?(?:\s*(?:x|\*|X)\s*10\s*\^\s*\{?[-+]?\d+\}?)?)"
)

ANSWER_UNITS = [
    "V/m",
    "N/C",
    "N m^2/C^2",
    "microF",
    "microC",
    "microJ",
    "uF",
    "uC",
    "uJ",
    "mF",
    "mC",
    "mJ",
    "nF",
    "nC",
    "nJ",
    "pF",
    "mH",
    "kHz",
    "Hz",
    "ohm",
    "Omega",
    "N",
    "V",
    "A",
    "F",
    "C",
    "J",
    "H",
    "T",
    "Wb",
    "W",
    "%",
    "cm",
    "m",
    "degree",
]


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def normalize_text(text: Any) -> str:
    value = str(text or "")
    value = re.sub(
        r"10([⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+)",
        lambda match: "10^" + "".join(SUPERSCRIPT_EXPONENTS.get(char, char) for char in match.group(1)),
        value,
    )
    value = value.translate(SUPERSCRIPT_TRANSLATION)
    replacements = {
        "\u00d7": "x",
        "\u2212": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u03bc": "u",
        "\u00b5": "u",
        "\u03a9": "ohm",
        "\u03c0": "pi",
        "\u03c6": "phi",
        "\u03b5": "epsilon",
        "\u221a": "sqrt",
        "\u2032": "'",
        "\u2019": "'",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def parse_number(token: str) -> float:
    legacy = normalize_text(token)
    legacy = re.sub(r"(\d(?:\.\d+)?)\s+\.\s*10\s*\^?\{?([-+]?\d+)\}?", r"\1x10^\2", legacy)
    value = legacy.replace(" ", "")
    bare_power = re.fullmatch(r"([-+]?)10\^\{?([-+]?\d+)\}?", value)
    if bare_power:
        sign = -1.0 if bare_power.group(1) == "-" else 1.0
        return sign * (10.0 ** int(bare_power.group(2)))
    match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)(?:x|\*|X)10\^?\{?([-+]?\d+)\}?", value)
    if match:
        return float(match.group(1)) * (10.0 ** int(match.group(2)))
    return float(value)


def strip_code_fence(text: str) -> str:
    value = str(text).strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    return value.strip()


def extract_json_object(text: str) -> dict[str, Any]:
    value = strip_code_fence(text)
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = value.find("{")
    if start < 0:
        raise ValueError("No JSON object start found")
    depth = 0
    in_string = False
    escape = False
    for pos in range(start, len(value)):
        char = value[pos]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                parsed = json.loads(value[start : pos + 1])
                if not isinstance(parsed, dict):
                    raise ValueError("Extracted JSON is not an object")
                return parsed
    raise ValueError("No complete JSON object found")


def numbered_cot(items: list[Any]) -> list[str]:
    result: list[str] = []
    for idx, item in enumerate(items or [], start=1):
        text = str(item).strip()
        text = re.sub(r"^Step\s*\d+\s*:\s*", "", text, flags=re.IGNORECASE)
        result.append(f"Step {idx}: {text}")
    return result


def compact_answer_text(answer: Any) -> str:
    text = normalize_text(answer)
    if not text:
        return "Unknown"
    if text.lower() == "unknown":
        return "Unknown"
    if not re.search(r"\b(is|equals|directed|towards|because|therefore|answer)\b", text, flags=re.IGNORECASE):
        return text

    unit_pattern = "|".join(re.escape(unit) for unit in sorted(ANSWER_UNITS, key=len, reverse=True))
    match = re.search(rf"({NUMBER_RE})\s*({unit_pattern})?\b", text, flags=re.IGNORECASE)
    if not match:
        return text
    value = match.group(1).strip()
    unit = (match.group(2) or "").strip()
    return f"{value} {unit}".strip()


def first_number(text: str) -> float | None:
    normalized = normalize_text(text).replace(",", "")
    match = re.search(NUMBER_RE, normalized)
    if not match:
        return None
    try:
        return parse_number(match.group(0))
    except ValueError:
        return None
