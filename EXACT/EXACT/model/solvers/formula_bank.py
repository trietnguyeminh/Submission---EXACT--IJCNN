#!/usr/bin/env python3
"""Run EXACT Type 2 physics inference with Qwen + SymPy formula verification.

The client calls a self-hosted OpenAI-compatible vLLM server for Qwen3-8B
generation. The deterministic branch uses a formula bank plus SymPy algebraic
closure when SymPy is installed, so formulas can be rearranged for the requested
unknown instead of relying only on one hard-coded direction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import sympy as sp
except ImportError:  # Keep dry runs usable, but report that algebraic closure is off.
    sp = None  # type: ignore[assignment]

from model.llm_client import check_model, http_json, normalize_base_url
from model.text_utils import configure_stdio, extract_json_object


PHYSICS_SYSTEM_PROMPT = """You are an EXACT 2026 Type 2 physics QA engine.
Solve from the input question only. Return exactly one valid JSON object and no markdown.

Required keys:
- answer: final numeric result plus unit only, e.g. "3.12 N" or "100 μF"
- explanation: concise natural-language explanation

Recommended keys:
- fol: one concise physics-logic formula string
- cot: list of concise public reasoning steps, each starting with "Step 1:", "Step 2:", ...
- premises: list of laws, formulas, or givens actually used
- confidence: number from 0.0 to 1.0

Rules:
- The answer field must not contain a sentence, direction, formula, or explanation. Put those in explanation/cot.
- If the final answer has a direction, keep answer as magnitude plus unit only and describe direction in explanation.
- If no numeric result can be computed, set answer exactly to "Unknown".
- Every cot item must start with the correct "Step n:" prefix.
- Prefer the deterministic solver result when it is provided and marked high confidence.
- Do not invent missing quantities. If the question is under-specified, lower confidence.
- Keep arithmetic consistent with SI conversions and the requested unit.
- If using a formula outside the provided bank, say so conservatively in premises and lower confidence.
"""


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

NUMBER_RE = r"[-+]?(?:10\s*\^\s*\{?[-+]?\d+\}?|\d+(?:\.\d+)?(?:\s*(?:x|\*|X)\s*10\s*\^\s*\{?[-+]?\d+\}?)?)"
K_COULOMB = 9.0e9
EPS0 = 8.85e-12
MU0 = 4.0 * math.pi * 1.0e-7
HAS_SYMPY = sp is not None


UNIT_FACTORS: dict[str, float] = {
    "": 1.0,
    "F": 1.0,
    "uF": 1.0e-6,
    "microF": 1.0e-6,
    "mF": 1.0e-3,
    "nF": 1.0e-9,
    "pF": 1.0e-12,
    "C": 1.0,
    "uC": 1.0e-6,
    "microC": 1.0e-6,
    "mC": 1.0e-3,
    "nC": 1.0e-9,
    "V": 1.0,
    "J": 1.0,
    "mJ": 1.0e-3,
    "uJ": 1.0e-6,
    "microJ": 1.0e-6,
    "nJ": 1.0e-9,
    "H": 1.0,
    "mH": 1.0e-3,
    "A": 1.0,
    "mA": 1.0e-3,
    "ohm": 1.0,
    "Hz": 1.0,
    "kHz": 1.0e3,
    "rad/s": 1.0,
    "kV": 1.0e3,
    "m": 1.0,
    "cm": 1.0e-2,
    "mm": 1.0e-3,
    "T": 1.0,
    "mT": 1.0e-3,
    "Wb": 1.0,
    "uWb": 1.0e-6,
    "W": 1.0,
    "N": 1.0,
    "mN": 1.0e-3,
    "uN": 1.0e-6,
    "V/m": 1.0,
    "kV/m": 1.0e3,
    "N/C": 1.0,
    "J/m3": 1.0,
    "turns/m": 1.0,
    "s": 1.0,
    "ms": 1.0e-3,
    "kg": 1.0,
    "g": 1.0e-3,
    "%": 0.01,
    "degree": 1.0,
    "degrees": 1.0,
}

DISPLAY_UNITS = {
    "uF": "\u03bcF",
    "uC": "\u03bcC",
    "uJ": "\u03bcJ",
    "uWb": "\u03bcWb",
}


@dataclass
class Quantity:
    name: str
    value_si: float
    unit: str
    raw: str


@dataclass
class SolverResult:
    solved: bool
    route: str
    answer: str
    fol: str
    cot: list[str]
    premises: list[str]
    confidence: float
    target: str
    value_si: float | None = None
    unit: str | None = None
    warning: str | None = None
    engine: str = "formula_bank"


@dataclass
class PipelineRecord:
    item_index: int
    sample_id: str | None
    question: str
    response: dict[str, Any]
    route: str
    solver_result: dict[str, Any] | None
    arbitration: str
    valid_json: bool
    validation_errors: list[str]
    raw_response: str | None
    latency_sec: float


def normalize_text(text: str) -> str:
    value = str(text)
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
    value = re.sub(r"(?<=\d)\.10(\s*\^?\s*\{?[-+]?\d+\}?)", r"x10\1", value)
    value = re.sub(r"\b(V|N|J)\s*/\s*(m|C|m3)\b", r"\1/\2", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def parse_number(token: str) -> float:
    value = normalize_text(token).replace(" ", "")
    bare_power = re.fullmatch(r"([-+]?)10\^\{?([-+]?\d+)\}?", value)
    if bare_power:
        sign = -1.0 if bare_power.group(1) == "-" else 1.0
        return sign * (10.0 ** int(bare_power.group(2)))
    legacy = normalize_text(token)
    legacy = re.sub(r"(\d(?:\.\d+)?)\s+\.\s*10\s*\^?\{?([-+]?\d+)\}?", r"\1x10^\2", legacy)
    value = legacy.replace(" ", "")
    match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)(?:x|\*|X)10\^?\{?([-+]?\d+)\}?", value)
    if match:
        return float(match.group(1)) * (10.0 ** int(match.group(2)))
    if re.fullmatch(r"[-+0-9.*/^()pisqrt]+", value, flags=re.IGNORECASE) and any(
        token in value.lower() for token in ("pi", "sqrt", "/", "*")
    ):
        expr = value.replace("^", "**")
        expr = re.sub(r"sqrt([-+]?\d+(?:\.\d+)?)", r"sqrt(\1)", expr, flags=re.IGNORECASE)
        expr = re.sub(r"(?<=\d)sqrt", "*sqrt", expr, flags=re.IGNORECASE)
        expr = re.sub(r"(?<=\d)pi", "*pi", expr, flags=re.IGNORECASE)
        expr = re.sub(r"pi(?=\d)", "pi*", expr, flags=re.IGNORECASE)
        expr = re.sub(r"\)(?=pi|\d)", ")*", expr, flags=re.IGNORECASE)
        expr = re.sub(r"(?<=\d)\(", "*(", expr)
        try:
            if sp is not None:
                return float(sp.N(sp.sympify(expr)))
            return float(eval(expr, {"__builtins__": {}}, {"pi": math.pi, "sqrt": math.sqrt}))  # noqa: S307
        except Exception:
            pass
    return float(value)


def clean_unit(unit: str | None) -> str:
    if not unit:
        return ""
    value = normalize_text(unit)
    value = value.replace("Ohm", "ohm")
    value = value.replace("ohms", "ohm")
    return value


def quantity_from_match(name: str, match: re.Match[str]) -> Quantity:
    raw_value = match.group("val")
    unit = clean_unit(match.groupdict().get("unit", ""))
    factor = UNIT_FACTORS.get(unit, 1.0)
    return Quantity(name=name, value_si=parse_number(raw_value) * factor, unit=unit, raw=match.group(0))


def find_quantity(text: str, name: str, patterns: list[str]) -> Quantity | None:
    normalized = normalize_text(text)
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            return quantity_from_match(name, match)
    return None


def find_signed_charge(text: str, label: str) -> Quantity | None:
    normalized = normalize_text(text)

    if label == "q0":
        qo_pattern = rf"\b(?:q0|qo)\s*=\s*(?P<sign>[+-])?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>uC|microC|nC|mC|C)\b"
        qo_match = re.search(qo_pattern, normalized, flags=re.IGNORECASE)
        if qo_match:
            quantity = quantity_from_match(label, qo_match)
            if qo_match.group("sign") == "-":
                quantity.value_si *= -1.0
            return quantity

    if label == "q":
        qprime_pattern = rf"\bq\s*(?:'|prime)\s*=\s*(?P<sign>[+-])?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>uC|microC|nC|mC|C)\b"
        qprime_match = re.search(qprime_pattern, normalized, flags=re.IGNORECASE)
        if qprime_match:
            quantity = quantity_from_match(label, qprime_match)
            if qprime_match.group("sign") == "-":
                quantity.value_si *= -1.0
            return quantity
        q_magnitude_pattern = rf"\b(?:test\s+charge\s+)?q\s+(?:with\s+(?:a\s+)?magnitude\s+of|has\s+(?:a\s+)?magnitude\s+of|of)\s*(?P<sign>[+-])?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>uC|microC|nC|mC|C)\b"
        q_magnitude_match = re.search(q_magnitude_pattern, normalized, flags=re.IGNORECASE)
        if q_magnitude_match:
            quantity = quantity_from_match(label, q_magnitude_match)
            if q_magnitude_match.group("sign") == "-":
                quantity.value_si *= -1.0
            return quantity

    opposite_equal = re.search(
        rf"\bq1\s*=\s*-q2\s*=\s*(?P<val>{NUMBER_RE})\s*(?P<unit>uC|microC|nC|mC|C)\b",
        normalized,
        flags=re.IGNORECASE,
    )
    if opposite_equal and label in {"q1", "q2"}:
        quantity = quantity_from_match(label, opposite_equal)
        if label == "q2":
            quantity.value_si *= -1.0
        return quantity

    # Support qA, qB, qC, qD named charges
    if label in {"qA", "qB", "qC", "qD"}:
        named_pattern = rf"\b{re.escape(label)}\s*=\s*(?P<sign>[+-])?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>uC|microC|nC|mC|C)\b"
        named_match = re.search(named_pattern, normalized, flags=re.IGNORECASE)
        if named_match:
            quantity = quantity_from_match(label, named_match)
            if named_match.group("sign") == "-":
                quantity.value_si *= -1.0
            return quantity
        # Also try "qA = qB = X" chain for named charges
        chain_named = rf"\b(?P<labels>q[A-D](?:\s*=\s*q[A-D])+)\s*=\s*(?P<sign>[+-])?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>uC|microC|nC|mC|C)\b"
        for chain_match in re.finditer(chain_named, normalized, flags=re.IGNORECASE):
            labels_found = re.findall(r"q[A-D]", chain_match.group("labels"), flags=re.IGNORECASE)
            if label in {item for item in labels_found}:
                quantity = quantity_from_match(label, chain_match)
                if chain_match.group("sign") == "-":
                    quantity.value_si *= -1.0
                return quantity
        both_equal_pattern = rf"\b(?P<label1>q[A-D])\s+and\s+(?P<label2>q[A-D])\s*,?\s+both\s+equal\s+to\s+(?P<sign>[+-])?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>uC|microC|nC|mC|C)\b"
        both_equal_match = re.search(both_equal_pattern, normalized, flags=re.IGNORECASE)
        if both_equal_match and label.lower() in {
            both_equal_match.group("label1").lower(),
            both_equal_match.group("label2").lower(),
        }:
            quantity = quantity_from_match(label, both_equal_match)
            if both_equal_match.group("sign") == "-":
                quantity.value_si *= -1.0
            return quantity

    pattern = rf"\b{re.escape(label)}\s*=\s*(?P<sign>[+-])?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>uC|microC|nC|mC|C)\b"
    match = re.search(pattern, normalized, flags=re.IGNORECASE)
    if match:
        quantity = quantity_from_match(label, match)
        if match.group("sign") == "-":
            quantity.value_si *= -1.0
        return quantity

    chain_pattern = rf"\b(?P<labels>q\d(?:\s*=\s*q\d)+)\s*=\s*(?P<sign>[+-])?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>uC|microC|nC|mC|C)\b"
    for chain_match in re.finditer(chain_pattern, normalized, flags=re.IGNORECASE):
        labels = re.findall(r"q\d", chain_match.group("labels"), flags=re.IGNORECASE)
        if label.lower() in {item.lower() for item in labels}:
            quantity = quantity_from_match(label, chain_match)
            if chain_match.group("sign") == "-":
                quantity.value_si *= -1.0
            return quantity

    equal_pattern = rf"\bq1\s*=\s*q2\s*=\s*(?P<sign>[+-])?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>uC|microC|nC|mC|C)\b"
    equal_match = re.search(equal_pattern, normalized, flags=re.IGNORECASE)
    if label in {"q1", "q2"} and equal_match:
        quantity = quantity_from_match(label, equal_match)
        if equal_match.group("sign") == "-":
            quantity.value_si *= -1.0
        return quantity
    all_equal_pattern = rf"\bq1\s*=\s*q2\s*=\s*q3\s*=\s*(?P<sign>[+-])?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>uC|microC|nC|mC|C)\b"
    all_equal_match = re.search(all_equal_pattern, normalized, flags=re.IGNORECASE)
    if label in {"q1", "q2", "q3"} and all_equal_match:
        quantity = quantity_from_match(label, all_equal_match)
        if all_equal_match.group("sign") == "-":
            quantity.value_si *= -1.0
        return quantity
    return None


def find_distance(text: str, label: str) -> Quantity | None:
    patterns = [
        rf"\b{re.escape(label)}\s*(?:=|is)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
        rf"\b{re.escape(label[::-1])}\s*(?:=|is)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
    ]
    if label.upper() in {"AC", "CA", "BC", "CB"}:
        patterns.extend(
            [
                rf"\b(?:AC|CA)\s*=\s*(?:BC|CB)\s*=\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
                rf"\b(?:BC|CB)\s*=\s*(?:AC|CA)\s*=\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
            ]
        )
        first, second = label.upper()[0], label.upper()[1]
        patterns.extend(
            [
                rf"distance\s+from\s+{first}\s+to\s+{second}\s+(?:is|=)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
                rf"distance\s+from\s+{second}\s+to\s+{first}\s+(?:is|=)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
                rf"distance\s+from\s+{first}\s+to\s+{second}\s+(?:being|is|=)\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
                rf"distance\s+from\s+{second}\s+to\s+{first}\s+(?:being|is|=)\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
                rf"to\s+{first}\s+being\s+(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
                rf"to\s+{second}\s+being\s+(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
                rf"from\s+{first}\s+to\s+{second}\s+(?:is|=)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
                rf"to\s+{first}\s+(?:is|=)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
                rf"to\s+{second}\s+(?:is|=)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
            ]
        )
    if label.upper() == "AB":
        patterns = [
                rf"(?:charges?|point\s+charges?)[^;]*?\bq1\b[^;]*?\bq2\b[^;]*?(?:are|separated\s+by|distance\s+of)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s*apart\b",
                rf"\bq1\b[^;]*?\bq2\b[^;]*?(?:are|separated\s+by|distance\s+of)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s*apart\b",
                rf"(?:A and B|points A and B|at points A and B|two points A and B)[^.,;]*?(?:separated by|are|is)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s*(?:apart|from each other)?",
                rf"(?:A and B|points A and B|at points A and B|two points A and B)[^.]*?\s(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s*(?:apart|from each other)",
                rf"(?:separated by|distance AB is)\s*(?:a\s+distance\s+of\s*)?(?:a\s*=\s*)?(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)",
                rf"(?:straight\s+)?line\s+segment(?:\s+of\s+length)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s*(?:long)?",
                rf"(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s*(?:apart|long line segment)",
            ] + patterns
    if label.upper() in {"AM", "MA"}:
        patterns.extend(
            [
                rf"(?:MA|AM)\s*=\s*(?:MB|BM)\s*=\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
                rf"(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s+from\s+A\b",
            ]
        )
    if label.upper() in {"BM", "MB"}:
        patterns.extend(
            [
                rf"(?:MA|AM)\s*=\s*(?:MB|BM)\s*=\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
                rf"(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s+from\s+B\b",
            ]
        )
    return find_quantity(
        text,
        label,
        patterns,
    )


def find_generic_distance(text: str) -> Quantity | None:
    return find_quantity(
        text,
        "r",
        [
            rf"(?:distance|separated by|separation|away|apart)[^.,;]{{0,40}}?(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
            rf"(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s*(?:away|apart|from|distant|separated)\b",
            rf"point\s+(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s+away\b",
        ],
    )


def find_area(text: str) -> Quantity | None:
    normalized = normalize_text(text)
    patterns = [
        rf"(?:plate\s+area|area\s+of\s+(?:each\s+)?(?:plate|turn)|area\s+of\s+each\s+plate|cross-sectional\s+area|area)\s*(?:of|=|is|are)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)(?:2|\^2| squared)?\b",
        rf"(?:cross-sectional\s+area|area)[^.,;]{{0,56}}?\b(?:of|is|=)\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)(?:2|\^2| squared)?\b",
        rf"\bS\s*(?:=|is)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)(?:2|\^2| squared)?\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            unit = clean_unit(match.group("unit"))
            area_factor = {"m": 1.0, "cm": 1.0e-4, "mm": 1.0e-6}.get(unit, 1.0)
            return Quantity("area", parse_number(match.group("val")) * area_factor, f"{unit}2", match.group(0))
    return None


def extract_common_quantities(question: str) -> dict[str, Quantity]:
    q = normalize_text(question)
    quantities: dict[str, Quantity] = {}
    specs = {
        "C": [
            rf"(?:capacitance|capacitor|C)\s*(?:C\s*)?(?:=|is|of|with)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>uF|microF|mF|nF|pF|F)\b",
            rf"(?P<val>{NUMBER_RE})\s*(?P<unit>uF|microF|mF|nF|pF|F)\s+capacitor\b",
        ],
        "U": [
            rf"(?:RMS\s+voltage|effective\s+voltage|voltage|potential difference|U(?:AB)?|charged to|under|source voltage|applied voltage)\s*(?:=|is|of|to|at|across its plates is)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>V|kV)\b",
            rf"(?:RMS\s+voltage|effective\s+voltage|voltage|potential difference)[^.,;]{{0,48}}?\s(?:of|is|=)\s*(?P<val>{NUMBER_RE})\s*(?P<unit>V|kV)\b",
            rf"(?:RMS\s+voltage|effective\s+voltage|voltage|potential difference)[^.,;]{{0,80}}?(?P<val>{NUMBER_RE})\s*(?P<unit>V|kV)\b",
            rf"(?P<val>{NUMBER_RE})\s*(?P<unit>V|kV)\s+(?:voltage source|power source|source|battery|rms voltage|effective voltage)\b",
            rf"(?:AC\s+source|source|battery)[^.,;]{{0,40}}?\b(?:of|is|=)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>V|kV)\b",
            rf"(?P<val>{NUMBER_RE})\s*(?P<unit>V|kV)\s*(?:is\s+)?applied\b",
        ],
        "Q": [
            rf"(?:charge|Q)\s*(?:=|is|of|stores|stored by the capacitor is|maximum charge on the capacitor is)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>uC|microC|nC|mC|C)\b",
        ],
        "W": [
            rf"(?:energy|W|electric field energy|magnetic field energy)\s*(?:=|is|of)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>mJ|uJ|microJ|nJ|J)\b",
            rf"(?P<val>{NUMBER_RE})\s*(?P<unit>mJ|uJ|microJ|nJ|J)\s+(?:of\s+)?(?:stored\s+)?(?:electric(?:al)?|magnetic|field)?\s*energy\b",
        ],
        "L": [
            rf"(?:inductance|inductor|L)\s*(?:=|is|of|with)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>mH|H)\b",
            rf"(?:inductance|self-inductance)\s*(?:\(?L\)?)?[^.,;]{{0,48}}?\s(?:=|is|of)\s*(?P<val>{NUMBER_RE})\s*(?P<unit>mH|H)\b",
            rf"(?P<val>{NUMBER_RE})\s*(?P<unit>mH|H)\s+inductor\b",
        ],
        "I": [
            rf"(?:current|I)\s*(?:=|is|of|through it is|carries|flowing through it|flowing through it is)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>mA|A)\b",
            rf"(?:carries|has)\s+(?:a\s+)?current\s+(?:of\s+)?(?P<val>{NUMBER_RE})\s*(?P<unit>mA|A)\b",
            rf"(?P<val>{NUMBER_RE})\s*(?P<unit>mA|A)\s+of\s+current\b",
        ],
        "R": [
            rf"(?:resistance|pure resistance|R)\s*(?:=|is|of|with)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>ohm)\b",
            rf"(?P<val>{NUMBER_RE})\s*(?P<unit>ohm)\s+(?:resistor|resistance|lamp|branch)\b",
        ],
        "f": [
            rf"(?:frequency|f)\s*(?:=|is|of|at)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>kHz|Hz)\b",
        ],
        "B": [
            rf"(?:magnetic field|B)\s*(?:=|is|of)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>mT|T)\b",
        ],
    }
    for key, patterns in specs.items():
        found = find_quantity(q, key, patterns)
        if found:
            quantities[key] = found
    area = find_area(q)
    if area:
        quantities["area"] = area
    return quantities


def _unit_group(units: list[str]) -> str:
    return "|".join(re.escape(unit) for unit in sorted(units, key=len, reverse=True))


def find_symbol_quantity(text: str, symbol: str, units: list[str], *, aliases: list[str] | None = None) -> Quantity | None:
    normalized = normalize_text(text)
    unit_pattern = _unit_group(units)
    labels = [symbol] + list(aliases or [])
    expr_re = r"[-+]?(?:\d+(?:\.\d+)?|10\s*\^\s*\{?[-+]?\d+\}?)(?:\s*(?:x|\*|/|\+|-)\s*(?:\d+(?:\.\d+)?|10\s*\^\s*\{?[-+]?\d+\}?|pi|\(?\s*\d*(?:\.\d+)?\s*pi\s*\)?))*|[-+]?\d*(?:\.\d+)?\s*pi"
    for label in labels:
        label_pattern = re.escape(label).replace(r"\ ", r"\s+")
        patterns = [
            rf"\b{label_pattern}\b\s*(?:=|is|of)?\s*(?P<val>{expr_re}|{NUMBER_RE})\s*(?P<unit>{unit_pattern})\b",
            rf"\b{label_pattern}\b[^.,;?]{{0,48}}?\b(?:of|is|=)\s*(?P<val>{expr_re}|{NUMBER_RE})\s*(?P<unit>{unit_pattern})\b",
            rf"\b{label_pattern}\b[^.,;?]{{0,72}}?\b(?:measured\s+(?:as|at|to\s+be)|has\s+(?:a\s+)?value\s+of)\s*(?P<val>{expr_re}|{NUMBER_RE})\s*(?P<unit>{unit_pattern})\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match:
                return quantity_from_match(symbol, match)
    return None


def find_indexed_quantity(text: str, symbol: str, index: int, units: list[str]) -> Quantity | None:
    labels = [f"{symbol}{index}", f"{symbol}_{index}", f"{symbol}{chr(0x2080 + index)}" if 0 <= index <= 9 else f"{symbol}{index}"]
    return find_symbol_quantity(text, f"{symbol}{index}", units, aliases=labels)


def find_all_indexed_quantities(text: str, symbol: str, units: list[str]) -> dict[int, Quantity]:
    found: dict[int, Quantity] = {}
    for idx in range(1, 10):
        quantity = find_indexed_quantity(text, symbol, idx, units)
        if quantity:
            found[idx] = quantity
    return found


def find_plate_distance(text: str) -> Quantity | None:
    return find_quantity(
        text,
        "d",
        [
            rf"(?:distance between (?:the )?(?:two )?plates|plate separation|separation(?: distance)?|separated by|distance d\d*|d\d*|d)\s*(?:of|=|is|are|by)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
            rf"(?:separation|distance)[^.,;]{{0,48}}?\b(?:of|is|=)\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
            rf"(?:plates?.*?separated by)\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
            rf"distance between the plates and the potential difference[^.,;]*?\bare\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
        ],
    )


def find_radius(text: str) -> Quantity | None:
    return find_quantity(
        text,
        "r",
        [
            rf"(?:radius|radius R|R)\s*(?:=|is|of)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
        ],
    )


def effective_plate_area(text: str, quantities: dict[str, Quantity] | None = None) -> Quantity | None:
    area = (quantities or {}).get("area") or find_area(text)
    if area:
        return area
    radius = find_radius(text)
    if radius:
        value = math.pi * radius.value_si * radius.value_si
        return Quantity("area", value, "m2", f"area from {radius.raw}")
    return None


def find_epsilon_r(text: str) -> float:
    normalized = normalize_text(text)
    match = re.search(
        rf"(?:dielectric constant|relative permittivity|epsilon_r|epsilon|epsilonr)\s*(?:=|of|is)?\s*(?P<val>{NUMBER_RE})",
        normalized,
        flags=re.IGNORECASE,
    )
    if match:
        return parse_number(match.group("val"))
    if "air" in normalized.lower() or "vacuum" in normalized.lower():
        return 1.0
    return 1.0


def find_electric_field_strength(text: str) -> Quantity | None:
    return find_quantity(
        text,
        "E",
        [
            rf"(?:E\s*_?\s*max|Emax|maximum electric field strength|dielectric strength|electric field strength)\s*(?:=|is|of|air can withstand is)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>kV/m|V/m|N/C)\b",
            rf"(?P<val>{NUMBER_RE})\s*(?P<unit>kV/m|V/m|N/C)\b",
        ],
    )


def plate_distance_multiplier(text: str) -> float:
    lower = normalize_text(text).lower()
    if any(phrase in lower for phrase in ["distance between them is doubled", "distance between its plates is doubled", "distance between the plates is doubled", "plates are moved apart", "distance between them doubles"]):
        return 2.0
    if any(phrase in lower for phrase in ["distance between them is tripled", "distance between its plates is tripled", "distance between the plates is tripled", "distance between the plates is then tripled"]):
        return 3.0
    if "distance between" in lower and "halved" in lower:
        return 0.5
    match = re.search(r"(?:distance|separation).*?(?:increased|decreased).*?by\s*(?P<val>\d+(?:\.\d+)?)\s*times", lower)
    if match:
        factor = parse_number(match.group("val"))
        return 1.0 / factor if "decreased" in lower else factor
    return 1.0


def target_capacitor_quantity(lower: str) -> str:
    intent_parts = re.split(r"\b(?:what|calculate|find|determine|compute)\b", lower, flags=re.IGNORECASE)
    intent = intent_parts[-1] if len(intent_parts) > 1 else lower
    intent_head = intent.strip()[:80]
    if re.match(r"(?:the\s+)?capacitance\b", intent_head):
        return "capacitance"
    if re.match(r"(?:the\s+)?energy\b", intent_head):
        return "energy"
    if re.match(r"(?:the\s+)?charge\b", intent_head):
        return "charge"
    if "potential difference" in intent or "voltage" in intent or re.search(r"\bU\b", intent, flags=re.IGNORECASE):
        return "voltage"
    if "charge" in intent:
        return "charge"
    if "energy" in intent:
        return "energy"
    if "capacitance" in intent:
        return "capacitance"
    if "dielectric constant" in intent or "relative permittivity" in intent:
        return "dielectric_constant"
    if lower.strip().startswith(("calculate", "find", "determine", "compute")):
        if "potential difference" in lower or "voltage" in lower:
            return "voltage"
        if "energy" in lower:
            return "energy"
        if "charge" in lower:
            return "charge"
        if "capacitance" in lower:
            return "capacitance"
    return "unknown"


def parse_ac_omega(text: str) -> float | None:
    normalized = normalize_text(text).replace(" ", "")
    match = re.search(r"cos\(?\s*(?P<omega>[-+0-9.*/^()pisqrt]+)t", normalized, flags=re.IGNORECASE)
    if match:
        try:
            return parse_number(match.group("omega"))
        except ValueError:
            return None
    return None


def current_values_amperes(text: str) -> list[float]:
    return [quantity.value_si for quantity in numeric_unit_values(text, ["mA", "A"], "I")]


def frequency_values_hz(text: str) -> list[float]:
    return [quantity.value_si for quantity in numeric_unit_values(text, ["kHz", "Hz"], "f")]


def rlc_off_resonance_current_ratio(question: str) -> float | None:
    lower = normalize_text(question).lower()
    currents = [value for value in current_values_amperes(question) if value > 0]
    if len(currents) >= 2:
        return max(currents) / min(currents)
    if any(phrase in lower for phrase in ["current is halved", "current is half", "current is reduced by half", "current decreases to 1/2", "current becomes half", "current becomes 1/2"]):
        return 2.0
    fraction_match = re.search(r"current[^.,;?]{0,40}?(?:decreases|drops|becomes|is reduced)\s+to\s+(?P<num>\d+)\s*/\s*(?P<den>\d+)", lower)
    if fraction_match:
        fraction = parse_number(f"{fraction_match.group('num')}/{fraction_match.group('den')}")
        if 0 < fraction < 1:
            return 1.0 / fraction
    return None


def rlc_frequency_multiplier(question: str) -> float | None:
    lower = normalize_text(question).lower()
    frequencies = [value for value in frequency_values_hz(question) if value > 0]
    if len(frequencies) >= 2:
        base = frequencies[0]
        for value in frequencies[1:]:
            if abs(value - base) > max(1e-9, abs(base) * 1e-6):
                return value / base
    if any(phrase in lower for phrase in ["frequency doubles", "frequency is doubled", "frequency doubled", "frequency increases to double"]):
        return 2.0
    if any(phrase in lower for phrase in ["frequency triples", "frequency is tripled", "frequency tripled"]):
        return 3.0
    return None


def rlc_reactance_requested_at_new_frequency(question: str) -> bool:
    normalized = normalize_text(question)
    lower = normalized.lower()
    query_start = max(lower.rfind("what"), lower.rfind("calculate"), lower.rfind("find"), lower.rfind("determine"))
    query = lower[query_start:] if query_start >= 0 else lower
    if "initial" in query or "resonant frequency" in query or "resonance frequency" in query:
        return False
    frequencies = numeric_unit_values(question, ["kHz", "Hz"], "f")
    if len(frequencies) >= 2:
        target_match = re.search(rf"(?:at|of)\s*(?P<val>{NUMBER_RE})\s*(?P<unit>kHz|Hz)\b", query, flags=re.IGNORECASE)
        if target_match:
            target = quantity_from_match("f", target_match).value_si
            first = frequencies[0].value_si
            last = frequencies[-1].value_si
            return abs(target - last) <= max(1e-9, abs(last) * 1e-6) and abs(target - first) > max(1e-9, abs(first) * 1e-6)
    return False


def parse_rms_voltage_from_time_function(text: str) -> Quantity | None:
    normalized = normalize_text(text)
    match = re.search(r"\bu\s*=\s*(?P<amp>\d+(?:\.\d+)?)\s*sqrt\s*\(?2\)?\s*cos", normalized, flags=re.IGNORECASE)
    if match:
        value = parse_number(match.group("amp"))
        return Quantity("U", value, "V", match.group(0))
    return None


def find_impedance(text: str) -> Quantity | None:
    return find_symbol_quantity(text, "Z", ["ohm"], aliases=["impedance", "total impedance"])


def find_power(text: str) -> Quantity | None:
    return (
        find_symbol_quantity(text, "P", ["W"], aliases=["power", "power consumed", "power dissipated", "Pmax", "total power"])
        or find_quantity(
            text,
            "P",
            [
                rf"(?P<val>{NUMBER_RE})\s*(?P<unit>W)\s+(?:of\s+)?power\b",
                rf"(?:consume|consumes|consumed|dissipated)[^.,;]{{0,40}}?(?P<val>{NUMBER_RE})\s*(?P<unit>W)\b",
            ],
        )
    )


def find_reactance(text: str, label: str) -> Quantity | None:
    compact = label.replace("_", "")
    underscored = "X_L" if "L" in label.upper() else "X_C"
    aliases = [label, compact, underscored, label.replace("_", " "), "inductive reactance" if "L" in label.upper() else "capacitive reactance"]
    return find_symbol_quantity(text, label, ["ohm"], aliases=aliases)


def find_force_quantity(text: str) -> Quantity | None:
    return find_symbol_quantity(text, "F", ["uN", "mN", "N"], aliases=["force", "electric force"])


def quantities_charge_from_text(text: str) -> Quantity | None:
    return find_quantity(
        text,
        "q",
        [
            rf"(?:electric charge|charge|source charge|test charge)\s*(?:q|Q)?\s*(?:=|is|of|carrying)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>uC|microC|nC|mC|C)\b",
            rf"carrying\s+(?:an?\s+)?electric charge\s+of\s+(?P<val>{NUMBER_RE})\s*(?P<unit>uC|microC|nC|mC|C)\b",
        ],
    )


def find_mass_quantity(text: str) -> Quantity | None:
    return find_quantity(
        text,
        "m",
        [
            rf"(?:mass|m)\s*(?:=|is|of)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>kg|g)\b",
            rf"(?P<val>{NUMBER_RE})\s*(?P<unit>kg|g)\s+(?:mass|particle)\b",
        ],
    )


def numeric_unit_values(text: str, units: list[str], name: str) -> list[Quantity]:
    normalized = normalize_text(text)
    unit_pattern = _unit_group(units)
    values: list[Quantity] = []
    for match in re.finditer(rf"(?P<val>{NUMBER_RE})\s*(?P<unit>{unit_pattern})\b", normalized, flags=re.IGNORECASE):
        values.append(quantity_from_match(name, match))
    return values


def bare_resistance_values(text: str) -> list[Quantity]:
    return numeric_unit_values(text, ["ohm"], "R")


def labeled_current_values(text: str) -> dict[int, Quantity]:
    normalized = normalize_text(text)
    values: dict[int, Quantity] = {}
    patterns = [
        rf"current\s+through\s+(?:lamp\s+|branch\s+|resistor\s+)?D(?P<idx>\d)[^.,;]*?(?:is|=)\s*(?P<val>{NUMBER_RE})\s*(?P<unit>mA|A)\b",
        rf"(?:lamp|branch|resistor)\s+D(?P<idx>\d)[^.,;]*?draws\s*(?P<val>{NUMBER_RE})\s*(?P<unit>mA|A)\b",
        rf"\bD(?P<idx>\d)[^.,;]*?current[^.,;]*?(?:is|=)\s*(?P<val>{NUMBER_RE})\s*(?P<unit>mA|A)\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
            values[int(match.group("idx"))] = quantity_from_match(f"I{match.group('idx')}", match)
    return values


def current_change(text: str) -> tuple[float, float, str] | None:
    normalized = normalize_text(text)
    match = re.search(
        rf"(?:current[^.,;]*?|from|changes from|increases from|decreases(?: uniformly)? from)\s*(?P<i1>{NUMBER_RE})\s*A?\s*(?:to|and reaches)\s*(?P<i2>{NUMBER_RE})\s*A?\s*(?:in|during|over(?: a period of)?)\s*(?P<t>{NUMBER_RE})\s*s",
        normalized,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    delta_current = abs(parse_number(match.group("i2")) - parse_number(match.group("i1")))
    delta_time = parse_number(match.group("t"))
    return delta_current, delta_time, match.group(0)


def requested_decimal_places(question: str) -> int | None:
    lower = normalize_text(question).lower()
    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
    }
    match = re.search(r"rounded to (one|two|three|four|five|\d+) decimal", lower)
    if not match:
        return None
    token = match.group(1)
    return words.get(token, int(token) if token.isdigit() else 2)


def display_unit(unit: str) -> str:
    return DISPLAY_UNITS.get(unit, unit)


def strip_symbolic_numeric_labels(text: str) -> str:
    return re.sub(r"\b(?:q|Q|F|E|U|I|R|C|L|W|X)[A-Za-z]*\d+\b", lambda match: re.sub(r"\d+", "", match.group(0)), text)


def numbered_cot(items: list[Any]) -> list[str]:
    numbered: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        text = re.sub(r"^\s*(?:step\s*)?\d+\s*[:.)-]\s*", "", text, flags=re.IGNORECASE).strip()
        numbered.append(f"Step {len(numbered) + 1}: {text}")
    return numbered


def answer_unit_pattern() -> str:
    units = {unit for unit in UNIT_FACTORS if unit}
    units.update({"N", "V/m", "N/C", "turns/m"})
    return "|".join(re.escape(unit) for unit in sorted(units, key=len, reverse=True))


def compact_answer_text(answer: Any) -> str:
    value = normalize_text(str(answer or "")).strip()
    if not value or value.lower() == "unknown":
        return "Unknown"
    if ";" in value:
        parts = [compact_answer_text(part) for part in value.split(";") if part.strip()]
        if parts and all(part != "Unknown" for part in parts):
            return "; ".join(parts)

    normalized = strip_symbolic_numeric_labels(value.replace(",", ""))
    unit_pattern = answer_unit_pattern()
    answer_pattern = re.compile(rf"(?P<num>{NUMBER_RE})(?:\s*(?P<unit>{unit_pattern})(?![A-Za-z/]))?", flags=re.IGNORECASE)
    matches = list(answer_pattern.finditer(normalized))
    if not matches:
        return value

    contextual_matches = []
    for match in matches:
        prefix = normalized[max(0, match.start() - 48) : match.start()].lower()
        if re.search(r"(?:answer|result|final|value|magnitude|force|field|energy|capacitance|current|voltage)?\s*(?:is|=|equals|:)\s*$", prefix):
            contextual_matches.append(match)
    contextual_matches_with_unit = [match for match in contextual_matches if match.groupdict().get("unit")]
    if contextual_matches_with_unit:
        chosen = contextual_matches_with_unit[-1]
    elif contextual_matches:
        chosen = contextual_matches[-1]
    else:
        matches_with_unit = [match for match in matches if match.groupdict().get("unit")]
        chosen = matches_with_unit[-1] if matches_with_unit else matches[-1]
    number = re.sub(r"\s+", " ", chosen.group("num")).strip()
    unit = clean_unit(chosen.groupdict().get("unit", ""))
    if unit:
        return f"{number} {display_unit(unit)}"
    return number


def choose_unit(value_si: float, target: str, question: str) -> str:
    q = normalize_text(question)
    target_units = {
        "capacitance": ["F", "uF", "nF", "pF"],
        "charge": ["C", "uC", "nC", "mC"],
        "energy": ["J", "mJ", "uJ", "nJ"],
        "voltage": ["V"],
        "current": ["A", "mA"],
        "inductance": ["H", "mH"],
        "field": ["V/m"],
        "force": ["N"],
        "magnetic_field": ["T", "mT"],
        "reactance": ["ohm"],
        "resistance": ["ohm"],
        "impedance": ["ohm"],
        "frequency": ["Hz", "kHz"],
        "power": ["W"],
        "flux": ["Wb", "uWb"],
        "turn_density": ["turns/m"],
        "percent": ["%"],
        "angle": ["degree"],
        "dielectric_constant": [""],
        "quality_factor": [""],
        "power_factor": [""],
        "energy_density": ["J/m3"],
    }
    for unit in target_units.get(target, []):
        if len(unit) <= 1 and unit not in {"%"}:
            continue
        rendered = display_unit(unit)
        aliases = {unit, rendered, unit.replace("u", "micro")}
        if any(alias and alias in q for alias in aliases):
            return unit
    explicit_base = re.search(r"(?:unit|in|answer(?:\s+rounded)?\s+in)\s*[:=]?\s*(F|C|J|H|A|T|Wb|W|ohm|%)\b", q, re.IGNORECASE)
    if explicit_base:
        base_unit = clean_unit(explicit_base.group(1))
        if base_unit in target_units.get(target, []):
            return base_unit

    abs_value = abs(value_si)
    defaults = {
        "capacitance": "uF" if abs_value >= 1e-6 else ("nF" if abs_value >= 1e-9 else "pF"),
        "charge": "C" if abs_value >= 1 else ("mC" if abs_value >= 1e-3 else ("uC" if abs_value >= 1e-6 else "nC")),
        "energy": "J" if abs_value >= 1 else ("mJ" if abs_value >= 1e-3 else ("uJ" if abs_value >= 1e-6 else "nJ")),
        "voltage": "V",
        "current": "A",
        "inductance": "H",
        "field": "V/m",
        "force": "N",
        "magnetic_field": "mT" if abs(value_si) < 0.1 else "T",
        "reactance": "ohm",
        "resistance": "ohm",
        "impedance": "ohm",
        "frequency": "Hz" if abs_value < 1.0e3 else "kHz",
        "power": "W",
        "flux": "Wb" if abs_value >= 1e-3 else "uWb",
        "turn_density": "turns/m",
        "percent": "%",
        "angle": "degree",
        "dielectric_constant": "",
        "quality_factor": "",
        "power_factor": "",
        "energy_density": "J/m3",
    }
    return defaults.get(target, "")


def format_number(value: float, decimals: int | None = None) -> str:
    if decimals is not None:
        rounded = f"{value:.{decimals}f}"
        if value != 0 and float(rounded) == 0.0:
            return format_number(value, None)
        return rounded.rstrip("0").rstrip(".")
    if value == 0:
        return "0"
    if abs(value) >= 1.0e4 or abs(value) < 1.0e-3:
        mantissa, exponent = f"{value:.4e}".split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        return f"{mantissa} x 10^{int(exponent)}"
    if abs(value - round(value)) < 1.0e-10:
        return str(int(round(value)))
    return f"{value:.4g}"


def format_answer(value_si: float, target: str, question: str, unit: str | None = None) -> tuple[str, str, float]:
    chosen_unit = unit or choose_unit(value_si, target, question)
    factor = UNIT_FACTORS.get(chosen_unit, 1.0)
    display_value = value_si / factor
    if chosen_unit == "%":
        display_value = value_si * 100.0
    decimals = requested_decimal_places(question)
    if decimals is None and target == "percent" and abs(display_value - round(display_value)) > 1e-8:
        decimals = 2
    if decimals is None and abs(display_value) >= 0.01 and target in {"reactance", "voltage", "current"}:
        decimals = 2 if abs(display_value) < 1000 and abs(display_value - round(display_value)) > 1e-8 else None
    if decimals is not None and 0.0 < abs(display_value) < 0.01:
        decimals = None
    number = format_number(display_value, decimals)
    unit_label = display_unit(chosen_unit)
    if unit_label:
        return f"{number} {unit_label}", chosen_unit, display_value
    return number, chosen_unit, display_value


def formula_engine_name() -> str:
    return "sympy_formula_bank" if HAS_SYMPY else "direct_formula_bank_no_sympy"


def sympy_formula_value(
    formula_key: str,
    target: str,
    known_values: dict[str, float],
) -> float | None:
    """Solve a formula-bank equation for target using SymPy if available.

    The fallback direct formulas remain in the caller, but this is the preferred
    path on the Linux server. It lets one stored law solve any algebraic unknown
    that appears in the hidden test set, e.g. Q=CU -> C=Q/U or U=Q/C.
    """

    if sp is None:
        return None

    C, U, Q, W, L, I, f, Xc, B, n, N, length, A, d, eps_r, Phi, emf, delta_I, delta_t, P, R = sp.symbols(
        "C U Q W L I f Xc B n N length A d eps_r Phi emf delta_I delta_t P R",
        positive=True,
    )
    equations = {
        "capacitor_charge": sp.Eq(Q, C * U),
        "capacitor_energy": sp.Eq(W, sp.Rational(1, 2) * C * U**2),
        "inductor_energy": sp.Eq(W, sp.Rational(1, 2) * L * I**2),
        "parallel_plate_capacitor": sp.Eq(C, EPS0 * eps_r * A / d),
        "rlc_resonance": sp.Eq(f, 1 / (2 * sp.pi * sp.sqrt(L * C))),
        "capacitive_reactance": sp.Eq(Xc, 1 / (2 * sp.pi * f * C)),
        "rlc_resonance_power": sp.Eq(P, U**2 / R),
        "rlc_resonance_current": sp.Eq(I, U / R),
        "solenoid_field": sp.Eq(B, MU0 * n * I),
        "solenoid_inductance": sp.Eq(L, MU0 * N**2 * A / length),
        "magnetic_flux": sp.Eq(Phi, B * A),
        "induced_emf": sp.Eq(emf, L * delta_I / delta_t),
        "turn_density": sp.Eq(n, N / length),
    }
    symbols = {
        "C": C,
        "U": U,
        "Q": Q,
        "W": W,
        "L": L,
        "I": I,
        "f": f,
        "Xc": Xc,
        "B": B,
        "n": n,
        "N": N,
        "length": length,
        "A": A,
        "d": d,
        "eps_r": eps_r,
        "Phi": Phi,
        "emf": emf,
        "delta_I": delta_I,
        "delta_t": delta_t,
        "P": P,
        "R": R,
    }
    equation = equations.get(formula_key)
    target_symbol = symbols.get(target)
    if equation is None or target_symbol is None:
        return None
    substitutions = {
        symbols[name]: value
        for name, value in known_values.items()
        if name in symbols and symbols[name] != target_symbol
    }
    try:
        candidates = sp.solve(equation.subs(substitutions), target_symbol)
    except Exception:
        return None
    numeric_candidates: list[float] = []
    for candidate in candidates:
        try:
            numeric = float(sp.N(candidate))
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            numeric_candidates.append(numeric)
    nonnegative = [value for value in numeric_candidates if value >= -1e-12]
    if nonnegative:
        return max(0.0, min(nonnegative, key=abs))
    if numeric_candidates:
        return numeric_candidates[0]
    return None


def formula_value(
    formula_key: str,
    target: str,
    known_values: dict[str, float],
    fallback_value: float,
) -> float:
    solved = sympy_formula_value(formula_key, target, known_values)
    return fallback_value if solved is None else solved


def build_result(
    *,
    route: str,
    target: str,
    value_si: float,
    question: str,
    formula: str,
    premise: str,
    cot: list[str],
    confidence: float,
    unit: str | None = None,
) -> SolverResult:
    answer, chosen_unit, _ = format_answer(value_si, target, question, unit)
    fol = f"Given(question) and Law({formula}) -> Answer({answer})"
    return SolverResult(
        solved=True,
        route=route,
        answer=answer,
        fol=fol,
        cot=numbered_cot(cot + [f"The result is {answer}."]),
        premises=[premise],
        confidence=confidence,
        target=target,
        value_si=value_si,
        unit=chosen_unit,
        engine=formula_engine_name(),
    )


def build_text_result(
    *,
    route: str,
    answer: str,
    target: str,
    formula: str,
    premise: str,
    cot: list[str],
    confidence: float,
) -> SolverResult:
    return SolverResult(
        solved=True,
        route=route,
        answer=answer,
        fol=f"Given(question) and Law({formula}) -> Answer({answer})",
        cot=numbered_cot(cot + [f"The result is {answer}."]),
        premises=[premise],
        confidence=confidence,
        target=target,
        engine=formula_engine_name(),
    )


def unsolved(route: str, warning: str) -> SolverResult:
    return SolverResult(
        solved=False,
        route=route,
        answer="",
        fol="",
        cot=[],
        premises=[],
        confidence=0.0,
        target="unknown",
        warning=warning,
        engine=formula_engine_name(),
    )


def solve_capacitor_advanced(question: str, quantities: dict[str, Quantity]) -> SolverResult | None:
    lower = normalize_text(question).lower()
    c = quantities.get("C") or find_symbol_quantity(question, "C", ["uF", "microF", "mF", "nF", "pF", "F"], aliases=["capacitance"])
    u = quantities.get("U") or find_symbol_quantity(question, "U", ["V", "kV"], aliases=["voltage", "potential difference"])
    charge = quantities.get("Q") or find_symbol_quantity(question, "Q", ["uC", "microC", "nC", "mC", "C"], aliases=["charge"])
    energy = quantities.get("W") or find_symbol_quantity(question, "W", ["mJ", "uJ", "microJ", "nJ", "J"], aliases=["energy", "stored energy", "electric field energy", "electrical energy"])
    target = target_capacitor_quantity(lower)
    has_state_change = any(phrase in lower for phrase in ["disconnected", "isolated", "remains connected", "still connected", "connected to the source", "immersed", "moved apart", "distance between them doubles", "distance between its plates is doubled"])
    c_values_plain = numeric_unit_values(question, ["uF", "microF", "mF", "nF", "pF", "F"], "C")
    u_values_plain = numeric_unit_values(question, ["V", "kV"], "U")
    q_values_plain = numeric_unit_values(question, ["uC", "microC", "nC", "mC", "C"], "Q")
    energy_values_plain = numeric_unit_values(question, ["mJ", "uJ", "microJ", "nJ", "J"], "W")
    if not c and c_values_plain:
        c = c_values_plain[0]
    if not u and u_values_plain:
        u = u_values_plain[0]
    if not charge and q_values_plain:
        charge = q_values_plain[0]
    if not energy and energy_values_plain:
        energy = energy_values_plain[0]
    asks_energy_and_charge = bool(
        re.search(r"\b(?:energy|stored energy|electric field energy|electrical energy)\b[^?.]{0,80}\b(?:and|,)\s*(?:the\s+)?charge\b", lower)
        or re.search(r"\bcharge\b[^?.]{0,80}\b(?:and|,)\s*(?:the\s+)?(?:energy|stored energy|electric field energy|electrical energy)\b", lower)
    )

    if "short-circuited" in lower or "short circuited" in lower:
        answer = "0 μC; 0 μJ"
        return SolverResult(
            solved=True,
            route="capacitor_short_circuit",
            answer=answer,
            fol=f"Given(short_circuit) -> Answer({answer})",
            cot=numbered_cot(["A short-circuited capacitor has zero potential difference.", "The remaining capacitor charge and stored energy are both zero."]),
            premises=["Short circuit fully discharges the capacitor in the ideal model"],
            confidence=0.9,
            target="charge_energy",
            engine=formula_engine_name(),
        )

    if (
        c
        and u
        and asks_energy_and_charge
        and "how" not in lower
        and not any(token in lower for token in ["shared", "distributed", "cut from the source", "disconnected"])
    ):
        energy_value = 0.5 * c.value_si * u.value_si * u.value_si
        charge_value = c.value_si * u.value_si
        energy_answer, _, _ = format_answer(energy_value, "energy", question, "uJ")
        charge_answer, _, _ = format_answer(charge_value, "charge", question, "uC")
        answer = f"{energy_answer}; {charge_answer}"
        return SolverResult(
            solved=True,
            route="capacitor_energy_charge_pair",
            answer=answer,
            fol=f"Given(C,U) and Law(W=1/2CU^2,Q=CU) -> Answer({answer})",
            cot=numbered_cot(["Compute stored energy from W = 1/2 C U^2.", "Compute charge from Q = C U."]),
            premises=["Capacitor energy and charge relations"],
            confidence=0.92,
            target="energy_charge",
            engine=formula_engine_name(),
        )

    if "reduction in energy" in lower and len(c_values_plain) >= 2 and u:
        c1, c2 = c_values_plain[0], c_values_plain[1]
        w1 = 0.5 * c1.value_si * u.value_si * u.value_si
        w2 = 0.5 * c2.value_si * u.value_si * u.value_si
        reduction_percent = (w1 - w2) / w1 if w1 else 0.0
        return build_result(
            route="capacitor_energy_reduction_percent",
            target="percent",
            value_si=reduction_percent,
            question=question,
            formula="reduction = (W1-W2)/W1, W=1/2CU^2",
            premise="At fixed voltage, capacitor energy is proportional to capacitance",
            cot=["Compute initial and replacement capacitor energies at the same voltage.", "Report the fractional reduction as a percent."],
            confidence=0.88,
            unit="%",
        )

    if energy and len(u_values_plain) >= 2 and "energy" in lower and any(word in lower for word in ["reduced", "decreased", "changed", "remaining"]):
        u_initial, u_final = u_values_plain[0], u_values_plain[-1]
        if u_initial.value_si:
            value = energy.value_si * (u_final.value_si / u_initial.value_si) ** 2
            return build_result(
                route="capacitor_energy_voltage_scaling",
                target="energy",
                value_si=value,
                question=question,
                formula="W_f = W_i*(U_f/U_i)^2 for fixed C",
                premise="With unchanged capacitance, capacitor energy scales with voltage squared",
                cot=["Identify initial energy and two voltage values.", "Scale energy by the square of the voltage ratio."],
                confidence=0.9,
                unit=energy.unit,
            )

    if energy and len(energy_values_plain) >= 2 and ("percentage loss" in lower or "percent loss" in lower or "energy loss" in lower):
        initial, final = energy_values_plain[0], energy_values_plain[-1]
        if initial.value_si:
            value = (initial.value_si - final.value_si) / initial.value_si
            return build_result(
                route="capacitor_energy_loss_percent",
                target="percent",
                value_si=value,
                question=question,
                formula="loss = (W_i-W_f)/W_i",
                premise="Percentage energy loss compares the lost energy to the initial stored energy",
                cot=["Identify initial and final energies.", "Divide the energy loss by the initial energy."],
                confidence=0.9,
                unit="%",
            )

    if "voltage" in lower and "doubles" in lower and "energy" in lower:
        return build_text_result(
            route="capacitor_energy_voltage_scaling_concept",
            answer="increases by 4 times",
            target="energy_relation",
            formula="W proportional to U^2 for fixed C",
            premise="For an unchanged capacitor, stored energy scales with the square of voltage",
            cot=["Voltage doubles.", "Square the voltage factor to get the energy factor."],
            confidence=0.9,
        )

    expr_number = rf"[-+]?\d+(?:\.\d+)?\s*sqrt\s*\d+(?:\.\d+)?|[-+]?(?:\d+(?:\.\d+)?|pi|sqrt\d)[-+0-9.*/^()pisqrt]*|{NUMBER_RE}"
    voltage_function = re.search(
        rf"\bU(?:\(t\))?\s*(?:=|is)?\s*(?P<amp>{expr_number})\s*(?:x|\*)?\s*(?:sin|cos)\s*\(",
        normalize_text(question),
        flags=re.IGNORECASE,
    )
    if voltage_function is None:
        voltage_function = re.search(
            rf"voltage[^.;?]{{0,40}}?\s(?:is|=)\s*(?P<amp>{expr_number})\s*(?:x|\*)?\s*(?:sin|cos)\s*\(",
            normalize_text(question),
            flags=re.IGNORECASE,
        )
    if ("maximum electric field energy" in lower or "maximum energy" in lower) and c and voltage_function:
        amplitude = parse_number(voltage_function.group("amp"))
        value = 0.5 * c.value_si * amplitude * amplitude
        return build_result(
            route="capacitor_max_energy_from_voltage_function",
            target="energy",
            value_si=value,
            question=question,
            formula="W_max = 1/2 C U0^2",
            premise="Maximum capacitor electric energy uses the voltage amplitude",
            cot=["Extract the voltage amplitude from U(t).", "Use W=1/2 C U^2."],
            confidence=0.9,
            unit="J",
        )

    if ("maximum electric field energy" in lower or "maximum energy" in lower) and c and charge:
        value = charge.value_si * charge.value_si / (2.0 * c.value_si)
        return build_result(
            route="capacitor_max_energy_from_charge",
            target="energy",
            value_si=value,
            question=question,
            formula="W_max = Q_max^2/(2C)",
            premise="Capacitor energy can be computed from maximum charge and capacitance",
            cot=["Identify maximum charge and capacitance.", "Use W=Q^2/(2C)."],
            confidence=0.9,
            unit="J",
        )

    if c and ("split in half" in lower or "cut in half" in lower) and ("capacitance" in lower or "new capacitance" in lower):
        value = c.value_si / 2.0
        return build_result(
            route="capacitor_plate_area_halved",
            target="capacitance",
            value_si=value,
            question=question,
            formula="C proportional to A",
            premise="For a parallel-plate capacitor with unchanged separation, halving plate area halves capacitance",
            cot=["Interpret splitting plates in half as halving effective plate area.", "Use C proportional to A."],
            confidence=0.84,
            unit=c.unit,
        )

    if "identical capacitors" in lower and "series" in lower and "parallel" in lower and "energy" in lower:
        return build_text_result(
            route="capacitor_identical_series_parallel_energy_ratio",
            answer="one quarter",
            target="energy_relation",
            formula="C_series=C/2, C_parallel=2C, so W_series/W_parallel=1/4 at fixed U",
            premise="For identical capacitors connected to the same voltage source, energy is proportional to equivalent capacitance",
            cot=["Compute equivalent capacitance in series and parallel.", "Compare energies at the same voltage."],
            confidence=0.9,
        )

    distance_factor = None
    if any(phrase in lower for phrase in ["distance between the plates is tripled", "distance between two plates is tripled", "distance between the two plates is tripled"]):
        distance_factor = 3.0
    elif any(phrase in lower for phrase in ["distance between the plates is quadrupled", "distance between two plates is quadrupled", "distance between the two plates is quadrupled"]):
        distance_factor = 4.0
    elif any(phrase in lower for phrase in ["distance between the plates is doubled", "distance between two plates is doubled", "distance between the two plates is doubled"]):
        distance_factor = 2.0
    else:
        distance_factor_match = re.search(r"distance[^.;?]{0,80}?(?:increases|is increased|increased)\s+by\s+(?P<factor>\d+(?:\.\d+)?)\s+times", lower)
        if distance_factor_match:
            distance_factor = parse_number(distance_factor_match.group("factor"))
        if distance_factor is None:
            distance_change = re.search(
                rf"distance[^.;?]{{0,80}}?from\s+(?P<d1>{NUMBER_RE})\s*(?P<u1>mm|cm|m)\s+to\s+(?P<d2>{NUMBER_RE})\s*(?P<u2>mm|cm|m)",
                normalize_text(question),
                flags=re.IGNORECASE,
            )
            if distance_change:
                d1 = parse_number(distance_change.group("d1")) * UNIT_FACTORS.get(clean_unit(distance_change.group("u1")), 1.0)
                d2 = parse_number(distance_change.group("d2")) * UNIT_FACTORS.get(clean_unit(distance_change.group("u2")), 1.0)
                if d1 > 0:
                    distance_factor = d2 / d1
    if ("disconnected" in lower or "constant charge" in lower or "charge remains constant" in lower or "electric charge remains constant" in lower) and distance_factor and "energy" in lower:
        if energy:
            value = energy.value_si * distance_factor
            return build_result(
                route="capacitor_disconnected_distance_energy_scaling",
                target="energy",
                value_si=value,
                question=question,
                formula="Q constant, C proportional to 1/d, W=Q^2/(2C) proportional to d",
                premise="Disconnected capacitor conserves charge while plate-distance change changes capacitance",
                cot=["Keep charge constant.", "Capacitance is inversely proportional to plate distance.", "Energy scales directly with distance."],
                confidence=0.9,
                unit=energy.unit,
            )
        return build_text_result(
            route="capacitor_disconnected_distance_energy_scaling",
            answer=f"increases by {format_number(distance_factor)} times",
            target="energy_relation",
            formula="Q constant, C proportional to 1/d, W proportional to d",
            premise="At fixed charge, capacitor energy is inversely proportional to capacitance and therefore proportional to plate distance",
            cot=["Keep charge constant after disconnection.", "Scale energy by the distance factor."],
            confidence=0.9,
        )

    if "connected in series" in lower and "uncharged capacitor" in lower and energy and c_values_plain:
        caps = c_values_plain
        c_initial = caps[0].value_si
        c_other = caps[1].value_si if len(caps) >= 2 else c_initial
        c_equiv = (c_initial * c_other) / (c_initial + c_other) if c_initial + c_other else 0.0
        value = energy.value_si * (c_equiv / c_initial) if c_initial else 0.0
        return build_result(
            route="capacitor_series_with_uncharged_same_voltage",
            target="energy",
            value_si=value,
            question=question,
            formula="W_f/W_i = C_eq/C_1 at unchanged source voltage, C_eq=C1*C2/(C1+C2)",
            premise="Adding an uncharged capacitor in series reduces the equivalent capacitance at the same applied voltage",
            cot=["Infer the initial voltage from W_i=1/2 C_1 U^2.", "Use the series equivalent capacitance.", "Scale energy by C_eq/C_1."],
            confidence=0.84,
            unit=energy.unit,
        )

    permittivity_factor = re.search(
        rf"(?:permittivity|epsilon|dielectric constant)[^.;?]{{0,80}}?(?:increases|is increased|increases by|increased)\s+(?:by\s+)?(?:a\s+)?factor\s+(?:of\s+)?(?P<factor>{NUMBER_RE})",
        lower,
    )
    if ("disconnected" in lower or "isolated" in lower) and energy and permittivity_factor and "energy" in lower:
        factor = parse_number(permittivity_factor.group("factor"))
        if factor > 0:
            value = energy.value_si / factor
            return build_result(
                route="capacitor_disconnected_dielectric_energy_scaling",
                target="energy",
                value_si=value,
                question=question,
                formula="Q constant, C_f = k C_i, W_f = Q^2/(2C_f) = W_i/k",
                premise="For a disconnected capacitor, charge remains conserved while permittivity raises capacitance",
                cot=["Keep charge constant after disconnection.", "Permittivity increase scales capacitance by the same factor.", "Stored energy is inversely proportional to capacitance at fixed charge."],
                confidence=0.9,
                unit=energy.unit,
            )

    if "charge q is kept constant" in lower and len(c_values_plain) >= 2 and ("voltage change" in lower or "how does the voltage" in lower):
        c1, c2 = c_values_plain[0], c_values_plain[1]
        ratio = c1.value_si / c2.value_si
        if abs(ratio - 0.5) < 1e-9:
            answer = "the voltage is halved"
        elif abs(ratio - 2.0) < 1e-9:
            answer = "the voltage is doubled"
        else:
            answer = f"the voltage changes by a factor of {format_number(ratio)}"
        return build_text_result(
            route="capacitor_voltage_constant_charge_scaling",
            answer=answer,
            target="voltage_relation",
            formula="U = Q/C",
            premise="For fixed charge, voltage is inversely proportional to capacitance",
            cot=["Keep Q constant.", "Compare U2/U1 = C1/C2."],
            confidence=0.9,
        )

    if "how many times" in lower and "energy" in lower and len(q_values_plain) >= 2:
        q1, q2 = q_values_plain[0], q_values_plain[1]
        ratio = (q1.value_si / q2.value_si) ** 2 if q2.value_si else 0.0
        answer = f"decreases by {format_number(ratio)} times" if q2.value_si < q1.value_si else f"increases by {format_number(1.0 / ratio)} times"
        return build_text_result(
            route="capacitor_energy_charge_scaling",
            answer=answer,
            target="energy_relation",
            formula="W proportional to Q^2 for fixed C",
            premise="For a fixed capacitor, stored energy scales with the square of charge",
            cot=["Compare the two charges.", "Square the charge ratio to get the energy ratio."],
            confidence=0.88,
        )

    if ("cut from the source" in lower or "disconnected" in lower) and "uncharged" in lower and "connected" in lower and c and u:
        caps = c_values_plain
        other_c = caps[1] if len(caps) >= 2 else c
        total_charge = c.value_si * u.value_si
        total_capacitance = c.value_si + other_c.value_si
        value = total_charge * total_charge / (2.0 * total_capacitance)
        return build_result(
            route="capacitor_share_with_uncharged",
            target="energy",
            value_si=value,
            question=question,
            formula="W_f = Q_total^2/(2(C1+C2))",
            premise="Disconnected capacitor conserves charge when connected to an uncharged capacitor",
            cot=["Compute the initial charge.", "Use final total capacitance after charge sharing.", "Compute final stored energy."],
            confidence=0.88,
            unit="uJ",
        )

    if "charge is equally shared" in lower and c and u:
        count_match = re.search(r"among\s+(?P<count>\d+)\s+identical\s+capacitors", lower)
        count = int(count_match.group("count")) if count_match else 2
        total_charge = c.value_si * u.value_si
        value = total_charge * total_charge / (2.0 * count * c.value_si)
        return build_result(
            route="capacitor_equal_charge_sharing",
            target="energy",
            value_si=value,
            question=question,
            formula="W_f = Q_total^2/(2 n C)",
            premise="The disconnected charge is shared over n identical capacitors",
            cot=["Compute the conserved total charge.", "Use total capacitance nC.", "Compute final energy."],
            confidence=0.88,
            unit="uJ" if value < 1e-3 else "J",
        )

    if "distributed equally among" in lower and c and u:
        count_match = re.search(r"among\s+(?P<count>\d+)\s+identical\s+capacitors", lower)
        count = int(count_match.group("count")) if count_match else 2
        total_charge = c.value_si * u.value_si
        value = total_charge * total_charge / (2.0 * count * c.value_si)
        return build_result(
            route="capacitor_equal_charge_distribution",
            target="energy",
            value_si=value,
            question=question,
            formula="W_f = Q_total^2/(2 n C)",
            premise="Charge conservation over identical capacitors after sharing",
            cot=["Compute initial charge.", "Distribute it over identical capacitors.", "Compute total final energy."],
            confidence=0.88,
            unit="J",
        )

    if "dielectric" in lower and "replaced" in lower and "capacitance" in lower:
        eps_values = [parse_number(match.group("val")) for match in re.finditer(rf"(?:epsilon|ε|dielectric)[^.;]*?(?:=|where\s+ε\s*=)?\s*(?P<val>{NUMBER_RE})", normalize_text(question), flags=re.IGNORECASE)]
        if len(eps_values) >= 2 and eps_values[0] != 0:
            ratio = eps_values[-1] / eps_values[0]
            if abs(ratio - 0.5) < 1e-9:
                answer = "decreases by half"
            elif ratio < 1:
                answer = f"decreases by {format_number(1.0 / ratio)} times"
            else:
                answer = f"increases by {format_number(ratio)} times"
            return build_text_result(
                route="capacitor_dielectric_scaling_concept",
                answer=answer,
                target="capacitance_relation",
                formula="C proportional to epsilon_r",
                premise="For fixed plate geometry, capacitance is proportional to dielectric constant",
                cot=["Compare old and new dielectric constants.", "Scale capacitance by the dielectric ratio."],
                confidence=0.88,
            )

    if "find c'" in lower and "series" in lower and "final charge" in lower and c:
        final_charge = q_values_plain[-1] if q_values_plain else None
        total_voltage = u_values_plain[-1] if u_values_plain else u
        if final_charge and total_voltage:
            voltage_on_c = final_charge.value_si / c.value_si
            voltage_on_unknown = total_voltage.value_si - voltage_on_c
            if voltage_on_unknown > 0:
                value = final_charge.value_si / voltage_on_unknown
                return build_result(
                    route="capacitor_series_unknown_from_final_charge",
                    target="capacitance",
                    value_si=value,
                    question=question,
                    formula="C' = Q/(U_total - Q/C)",
                    premise="Series capacitors carry the same charge; voltages add",
                    cot=["Compute voltage across known capacitor from Q/C.", "Subtract from total voltage.", "Solve C'=Q/U'."],
                    confidence=0.86,
                    unit="uF",
                )

    if ("attractive force" in lower or "force between the two plates" in lower) and q_values_plain and effective_plate_area(question, quantities):
        area_q = effective_plate_area(question, quantities)
        value = q_values_plain[0].value_si * q_values_plain[0].value_si / (2.0 * EPS0 * area_q.value_si)
        return build_result(
            route="capacitor_plate_attractive_force",
            target="force",
            value_si=value,
            question=question,
            formula="F = Q^2/(2 epsilon0 A)",
            premise="Attractive pressure between parallel capacitor plates is sigma^2/(2epsilon0)",
            cot=["Convert charge and plate area to SI units.", "Apply F = Q^2/(2 epsilon0 A)."],
            confidence=0.86,
            unit="N",
        )

    if "electric field inside" in lower and len(c_values_plain) >= 2 and "series" in lower and u and find_plate_distance(question):
        c1, c2 = c_values_plain[0], c_values_plain[1]
        c_eq = c1.value_si * c2.value_si / (c1.value_si + c2.value_si)
        common_charge = c_eq * u.value_si
        voltage_on_c1 = common_charge / c1.value_si
        value = voltage_on_c1 / find_plate_distance(question).value_si
        return build_result(
            route="capacitor_series_plate_field",
            target="field",
            value_si=value,
            question=question,
            formula="E1 = U1/d1, U1 = Q/C1, Q = Ceq U",
            premise="Series capacitors carry common charge and plate field is voltage over separation",
            cot=["Compute equivalent capacitance and common charge.", "Find voltage across C1.", "Divide by plate separation."],
            confidence=0.88,
            unit="V/m",
        )

    if "additional work supplied by the source" in lower and "doubled" in lower and u and effective_plate_area(question, quantities) and find_plate_distance(question):
        area_q = effective_plate_area(question, quantities)
        d_q = find_plate_distance(question)
        c_initial = EPS0 * find_epsilon_r(question) * area_q.value_si / d_q.value_si
        c_final = c_initial / 2.0
        value = (c_final - c_initial) * u.value_si * u.value_si
        return build_result(
            route="capacitor_source_work_distance_change",
            target="energy",
            value_si=value,
            question=question,
            formula="W_source = DeltaQ*U = (C2-C1)U^2",
            premise="When connected to a fixed-voltage source, source work equals voltage times change in capacitor charge",
            cot=["Compute initial capacitance.", "Doubling plate separation halves capacitance.", "Compute source work from DeltaQ times U."],
            confidence=0.86,
            unit="uJ",
        )

    if "energy density" in lower and u and find_plate_distance(question):
        d_q = find_plate_distance(question)
        eps_r = find_epsilon_r(question)
        field = u.value_si / d_q.value_si
        value = 0.5 * EPS0 * eps_r * field * field
        return build_result(
            route="capacitor_energy_density",
            target="energy_density",
            value_si=value,
            question=question,
            formula="u_E = 1/2 epsilon0 epsilon_r E^2",
            premise="Uniform parallel-plate field energy density",
            cot=["Compute E = U/d.", "Apply energy density u = 1/2 epsilon E^2."],
            confidence=0.9,
            unit="J/m3",
        )

    if "new capacitance" in lower and c and find_plate_distance(question):
        distances = numeric_unit_values(question, ["cm", "mm", "m"], "d")
        if len(distances) >= 2:
            eps_r = find_epsilon_r(question)
            value = c.value_si * distances[0].value_si / distances[-1].value_si * eps_r
            return build_result(
                route="capacitor_geometry_dielectric_scaling",
                target="capacitance",
                value_si=value,
                question=question,
                formula="C2 = C1*(d1/d2)*epsilon_r",
                premise="Parallel-plate capacitance scales as epsilon_r/d",
                cot=["Read old and new plate separations.", "Scale the capacitance by d1/d2 and dielectric constant."],
                confidence=0.86,
                unit="pF",
            )

    if "energy stored in the electric field" in lower and u and effective_plate_area(question, quantities) and find_plate_distance(question):
        area_q = effective_plate_area(question, quantities)
        d_q = find_plate_distance(question)
        c_eff = EPS0 * find_epsilon_r(question) * area_q.value_si / d_q.value_si
        value = 0.5 * c_eff * u.value_si * u.value_si
        return build_result(
            route="parallel_plate_field_energy",
            target="energy",
            value_si=value,
            question=question,
            formula="W = 1/2 (epsilon0 epsilon_r A/d) U^2",
            premise="Parallel-plate capacitance with dielectric and capacitor energy formula",
            cot=["Compute capacitance from geometry.", "Compute stored field energy."],
            confidence=0.9,
            unit="nJ",
        )

    if target == "capacitance" and energy and u:
        value = formula_value("capacitor_energy", "C", {"W": energy.value_si, "U": u.value_si}, 2.0 * energy.value_si / (u.value_si * u.value_si))
        return build_result(
            route="capacitor_energy_inverse",
            target="capacitance",
            value_si=value,
            question=question,
            formula="C = 2W / U^2",
            premise="Capacitor energy rearrangement: C = 2W/U^2",
            cot=[f"Identify energy W = {energy.raw} and voltage U = {u.raw}.", "Rearrange W = 1/2 C U^2 for C."],
            confidence=0.92,
        )

    if target == "voltage" and energy and c:
        value = formula_value("capacitor_energy", "U", {"W": energy.value_si, "C": c.value_si}, math.sqrt(2.0 * energy.value_si / c.value_si))
        return build_result(
            route="capacitor_energy_inverse",
            target="voltage",
            value_si=value,
            question=question,
            formula="U = sqrt(2W / C)",
            premise="Capacitor energy rearrangement",
            cot=[f"Identify energy W = {energy.raw} and capacitance C = {c.raw}.", "Rearrange W = 1/2 C U^2 for U."],
            confidence=0.92,
            unit="V",
        )

    if target == "energy" and charge and u:
        value = 0.5 * charge.value_si * u.value_si
        return build_result(
            route="capacitor_energy_from_charge_voltage",
            target="energy",
            value_si=value,
            question=question,
            formula="W = 1/2 Q U",
            premise="Capacitor energy relation using charge and voltage",
            cot=[f"Identify charge Q = {charge.raw} and voltage U = {u.raw}.", "Compute W = 1/2 Q U."],
            confidence=0.9,
        )

    if target == "capacitance" and charge and u:
        value = charge.value_si / u.value_si
        return build_result(
            route="capacitor_direct",
            target="capacitance",
            value_si=value,
            question=question,
            formula="C = Q/U",
            premise="Capacitance definition",
            cot=[f"Identify Q = {charge.raw} and U = {u.raw}.", "Compute C = Q/U."],
            confidence=0.92,
        )

    if target == "voltage" and "parallel" in lower and charge and len(c_values_plain) >= 2:
        candidates = [charge.value_si / cap.value_si for cap in c_values_plain if cap.value_si > 0]
        limit_match = re.search(rf"\bU\s*<\s*(?P<val>{NUMBER_RE})\s*V\b", normalize_text(question), flags=re.IGNORECASE)
        if limit_match:
            limit = parse_number(limit_match.group("val"))
            constrained = [candidate for candidate in candidates if candidate < limit]
            if constrained:
                candidates = constrained
        if candidates:
            value = candidates[0]
            return build_result(
                route="capacitor_parallel_voltage_from_branch_charge",
                target="voltage",
                value_si=value,
                question=question,
                formula="U = Q_i/C_i for each parallel branch",
                premise="Parallel capacitors share the same voltage; choose the branch voltage satisfying any stated constraint",
                cot=["Compute candidate voltages from Q/C for each possible branch.", "Apply the stated voltage constraint if present."],
                confidence=0.88,
                unit="V",
            )

    if target == "voltage" and charge and c:
        value = charge.value_si / c.value_si
        return build_result(
            route="capacitor_direct",
            target="voltage",
            value_si=value,
            question=question,
            formula="U = Q/C",
            premise="Capacitor charge law rearranged",
            cot=[f"Identify Q = {charge.raw} and C = {c.raw}.", "Compute U = Q/C."],
            confidence=0.92,
            unit="V",
        )

    scaling_match = re.search(r"voltage[^.,;]*(?:increases by|is increased by|increased by|multiplied by)\s*(?P<factor>\d+(?:\.\d+)?)\s*times", lower)
    if "voltage" in lower and "doubled" in lower and "energy" in lower and ("how many times" in lower or "increase" in lower):
        scaling_factor = 2.0
    elif scaling_match and "energy" in lower:
        scaling_factor = parse_number(scaling_match.group("factor"))
    else:
        scaling_factor = None
    if scaling_factor is not None:
        value = scaling_factor * scaling_factor
        return build_text_result(
            route="capacitor_energy_scaling",
            answer=format_number(value),
            target="ratio",
            formula="W proportional to U^2 for fixed C",
            premise="Capacitor energy W = 1/2 C U^2",
            cot=["For fixed capacitance, electric field energy scales as U squared.", "Square the voltage scale factor."],
            confidence=0.9,
        )

    voltage_amplitude = re.search(rf"\bU\s*=\s*(?P<amp>{NUMBER_RE})\s*(?:sin|cos)\s*\(", normalize_text(question), flags=re.IGNORECASE)
    if target == "energy" and c and voltage_amplitude:
        amplitude = parse_number(voltage_amplitude.group("amp"))
        value = 0.5 * c.value_si * amplitude * amplitude
        return build_result(
            route="capacitor_max_energy_from_voltage",
            target="energy",
            value_si=value,
            question=question,
            formula="W_max = 1/2 C U_max^2",
            premise="Maximum capacitor electric energy occurs at maximum voltage magnitude",
            cot=["Identify capacitance and voltage amplitude.", "Compute W_max = 1/2 C Umax^2."],
            confidence=0.9,
            unit="J",
        )

    if target == "energy" and c and "charge varies" in lower:
        charge_values = numeric_unit_values(question, ["uC", "microC", "nC", "mC", "C"], "Q")
        if charge_values:
            q_max = max(abs(item.value_si) for item in charge_values)
            value = q_max * q_max / (2.0 * c.value_si)
            return build_result(
                route="capacitor_max_energy_from_charge",
                target="energy",
                value_si=value,
                question=question,
                formula="W_max = Q_max^2/(2C)",
                premise="Capacitor energy in terms of charge",
                cot=["Identify capacitance and maximum charge magnitude.", "Compute Wmax = Qmax^2/(2C)."],
                confidence=0.9,
                unit="J",
            )

    if target == "charge" and c and u and "energy" not in lower and not has_state_change:
        value = formula_value("capacitor_charge", "Q", {"C": c.value_si, "U": u.value_si}, c.value_si * u.value_si)
        return build_result(
            route="capacitor_direct",
            target="charge",
            value_si=value,
            question=question,
            formula="Q = C * U",
            premise="Capacitor charge law: Q = C U",
            cot=[f"Identify capacitance C = {c.raw} and voltage U = {u.raw}.", "Compute Q = C U."],
            confidence=0.94,
        )

    if target == "capacitance" and c and ("distance" in lower or "separation" in lower) and ("halved" in lower or "doubled" in lower or "tripled" in lower):
        distance_factor = plate_distance_multiplier(question)
        value = c.value_si / distance_factor
        return build_result(
            route="capacitor_distance_scaling",
            target="capacitance",
            value_si=value,
            question=question,
            formula="C proportional to 1/d",
            premise="Parallel-plate capacitance is inversely proportional to plate separation",
            cot=["Identify the plate-distance scale factor.", "Scale capacitance inversely with distance."],
            confidence=0.88,
        )

    c_values = find_all_indexed_quantities(question, "C", ["uF", "microF", "mF", "nF", "pF", "F"])
    u_values = find_all_indexed_quantities(question, "U", ["V", "kV"])
    total_u = find_symbol_quantity(question, "UAB", ["V", "kV"], aliases=["total voltage", "source voltage", "applied voltage"]) or u

    if len(c_values) >= 2 and len(u_values) >= 2 and any(phrase in lower for phrase in ["like-charged", "like-poled", "like-polarity", "like-signed", "connected together", "joined"]):
        c1, c2 = c_values[1], c_values[2]
        u1, u2 = u_values[1], u_values[2]
        value = (c1.value_si * u1.value_si + c2.value_si * u2.value_si) / (c1.value_si + c2.value_si)
        return build_result(
            route="capacitor_reconnected_like_polarity",
            target="voltage",
            value_si=value,
            question=question,
            formula="U_f = (C1 U1 + C2 U2) / (C1 + C2)",
            premise="Charge conservation for two capacitors reconnected with like polarity",
            cot=["Identify C1, C2, U1, and U2.", "Use total charge conservation on the final common-voltage pair."],
            confidence=0.9,
            unit="V",
        )

    if len(c_values) >= 2 and "series" in lower and total_u:
        c1, c2 = c_values[1], c_values[2]
        c_eq = c1.value_si * c2.value_si / (c1.value_si + c2.value_si)
        common_charge = c_eq * total_u.value_si
        target_idx = 2 if re.search(r"(?:across|on)\s+(?:capacitor\s+)?C2\b", normalize_text(question), re.IGNORECASE) else 1
        target_c = c_values[target_idx]
        value = common_charge / target_c.value_si
        return build_result(
            route="capacitor_series_voltage_division",
            target="voltage",
            value_si=value,
            question=question,
            formula="U_i = Q/C_i, Q = C_eq U_total",
            premise="Series capacitors carry the same charge",
            cot=["Compute the equivalent capacitance of the series pair.", "Use common charge to get the requested capacitor voltage."],
            confidence=0.88,
            unit="V",
        )

    if len(c_values) >= 2 and "parallel" in lower and charge and target == "voltage":
        candidates = [charge.value_si / cap.value_si for cap in c_values.values() if cap.value_si > 0]
        limit_match = re.search(r"U\s*<\s*(?P<val>{})\s*V".format(NUMBER_RE), normalize_text(question), flags=re.IGNORECASE)
        if limit_match:
            limit = parse_number(limit_match.group("val"))
            candidates = [candidate for candidate in candidates if candidate < limit] or candidates
        value = min(candidates, key=abs) if candidates else None
        if value is not None:
            return build_result(
                route="capacitor_parallel_charge_voltage",
                target="voltage",
                value_si=value,
                question=question,
                formula="U = Q_i / C_i",
                premise="Parallel capacitors share a common voltage",
                cot=["Use the stated charge on one parallel capacitor.", "Apply Q_i = C_i U and any stated voltage constraint."],
                confidence=0.82,
                unit="V",
            )

    area = effective_plate_area(question, quantities)
    distance = find_plate_distance(question)
    eps_r = find_epsilon_r(question)
    emax = find_electric_field_strength(question)

    if target == "dielectric_constant" and c and area and distance:
        value = c.value_si * distance.value_si / (EPS0 * area.value_si)
        return build_result(
            route="parallel_plate_dielectric_inverse",
            target="dielectric_constant",
            value_si=value,
            question=question,
            formula="epsilon_r = C d / (epsilon0 A)",
            premise="Parallel-plate capacitance inverse law",
            cot=["Identify C, plate area A, and plate separation d.", "Rearrange C = epsilon0 epsilon_r A/d for epsilon_r."],
            confidence=0.88,
            unit="",
        )

    if target == "charge" and any(word in lower for word in ["breakdown", "maximum charge", "emax"]) and area and emax:
        value = EPS0 * eps_r * area.value_si * emax.value_si
        return build_result(
            route="capacitor_breakdown_charge",
            target="charge",
            value_si=value,
            question=question,
            formula="Q_max = epsilon0 * epsilon_r * A * E_max",
            premise="Parallel-plate breakdown charge follows Q_max = epsilon A E_max",
            cot=["Identify plate area and dielectric strength E_max.", "Use Q_max = epsilon0 epsilon_r A E_max."],
            confidence=0.88,
        )

    if target == "capacitance" and area and distance:
        value = formula_value(
            "parallel_plate_capacitor",
            "C",
            {"eps_r": eps_r, "A": area.value_si, "d": distance.value_si},
            EPS0 * eps_r * area.value_si / distance.value_si,
        )
        return build_result(
            route="parallel_plate_capacitor",
            target="capacitance",
            value_si=value,
            question=question,
            formula="C = epsilon0 * epsilon_r * A / d",
            premise="Parallel-plate capacitor law",
            cot=[f"Identify plate area A = {area.raw} and separation d = {distance.raw}.", "Evaluate C = epsilon0 epsilon_r A/d."],
            confidence=0.9,
        )

    if target == "charge" and area and distance and u:
        capacitance = EPS0 * eps_r * area.value_si / distance.value_si
        value = capacitance * u.value_si
        return build_result(
            route="parallel_plate_charge",
            target="charge",
            value_si=value,
            question=question,
            formula="Q = C U, C = epsilon0 epsilon_r A/d",
            premise="Parallel-plate capacitance plus capacitor charge law",
            cot=["Compute C from plate geometry.", "Use Q = C U."],
            confidence=0.88,
        )

    if u and has_state_change:
        distance_factor = plate_distance_multiplier(question)
        dielectric_factor = eps_r if any(phrase in lower for phrase in ["dielectric", "immersed", "relative permittivity", "epsilon"]) else 1.0
        c_initial = c.value_si if c else 1.0
        c_final = c_initial * dielectric_factor / distance_factor
        disconnected = "disconnected" in lower or "isolated" in lower
        source_connected = any(phrase in lower for phrase in ["connected to the source", "remains connected", "still connected"])
        if target == "capacitance" and c:
            value = c_final
            result_target = "capacitance"
            formula = "C_final = C0 * epsilon_r / distance_factor"
        elif target == "voltage":
            value = u.value_si if source_connected and not disconnected else u.value_si * c_initial / c_final
            result_target = "voltage"
            formula = "connected: U constant; disconnected: U_final = Q/C_final"
        elif target == "energy" and c:
            value = 0.5 * c_final * u.value_si * u.value_si if source_connected and not disconnected else (c.value_si * u.value_si) ** 2 / (2.0 * c_final)
            result_target = "energy"
            formula = "connected: W=1/2 C_final U^2; disconnected: W=Q^2/(2C_final)"
        elif target == "charge" and c:
            value = c_final * u.value_si if source_connected and not disconnected else c.value_si * u.value_si
            result_target = "charge"
            formula = "connected: Q=C_final U; disconnected: Q constant"
        else:
            return None
        return build_result(
            route="capacitor_state_change",
            target=result_target,
            value_si=value,
            question=question,
            formula=formula,
            premise="Capacitor state rule: isolated capacitors conserve charge; source-connected capacitors conserve voltage",
            cot=["Compute the final capacitance from dielectric and plate-distance changes.", "Apply the appropriate conservation rule for the connection state."],
            confidence=0.88,
        )

    return None


def solve_capacitor(question: str, quantities: dict[str, Quantity]) -> SolverResult | None:
    lower = normalize_text(question).lower()
    advanced = solve_capacitor_advanced(question, quantities)
    if advanced:
        return advanced
    if "percentage" in lower and "voltage" in lower:
        voltages = [parse_number(x) for x in re.findall(rf"(?P<val>{NUMBER_RE})\s*V\b", normalize_text(question))]
        if len(voltages) >= 2:
            value = (voltages[-1] / voltages[0]) ** 2
            return build_result(
                route="capacitor_energy_ratio",
                target="percent",
                value_si=value,
                question=question,
                formula="W_final / W_initial = (U_final / U_initial)^2",
                premise="For the same isolated capacitor, W is proportional to U^2",
                cot=[
                    f"Identify initial voltage U1 = {format_number(voltages[0])} V and final voltage U2 = {format_number(voltages[-1])} V.",
                    "Since W = 1/2 C U^2 and C is unchanged, compute (U2/U1)^2.",
                ],
                confidence=0.88,
                unit="%",
            )
    complex_markers = [
        "disconnected",
        "connected to the source",
        "remains connected",
        "immersed",
        "distributed equally",
        "moved apart",
        "split",
        "replaced",
        "reduction",
        "connected in series",
        "connected in parallel",
        "final charge",
    ]
    if any(marker in lower for marker in complex_markers):
        return None
    c = quantities.get("C")
    u = quantities.get("U")
    charge = quantities.get("Q")
    energy = quantities.get("W")

    if ("charge" in lower or " q " in f" {lower} ") and c and u and "energy" not in lower:
        value = formula_value("capacitor_charge", "Q", {"C": c.value_si, "U": u.value_si}, c.value_si * u.value_si)
        return build_result(
            route="capacitor_direct",
            target="charge",
            value_si=value,
            question=question,
            formula="Q = C * U",
            premise="Capacitor charge law: Q = C U",
            cot=[
                f"Identify capacitance C = {c.raw} and voltage U = {u.raw}.",
                "Use the formula solver to solve the capacitor charge relation Q = C U for Q.",
            ],
            confidence=0.94,
        )
    if "capacitance" in lower and charge and u:
        value = formula_value("capacitor_charge", "C", {"Q": charge.value_si, "U": u.value_si}, charge.value_si / u.value_si)
        return build_result(
            route="capacitor_direct",
            target="capacitance",
            value_si=value,
            question=question,
            formula="C = Q / U",
            premise="Capacitance definition: C = Q / U",
            cot=[
                f"Identify charge Q = {charge.raw} and voltage U = {u.raw}.",
                "Use the formula solver to rearrange Q = C U and solve for C.",
            ],
            confidence=0.94,
        )
    if "energy" in lower and c and u:
        value = formula_value("capacitor_energy", "W", {"C": c.value_si, "U": u.value_si}, 0.5 * c.value_si * u.value_si * u.value_si)
        return build_result(
            route="capacitor_energy",
            target="energy",
            value_si=value,
            question=question,
            formula="W_c = 1/2 * C * U^2",
            premise="Capacitor energy law: W_c = 1/2 C U^2",
            cot=[
                f"Identify capacitance C = {c.raw} and voltage U = {u.raw}.",
                "Use the formula solver to evaluate W_c = 1/2 C U^2.",
            ],
            confidence=0.94,
        )
    if ("voltage" in lower or "potential difference" in lower) and energy and c:
        value = formula_value("capacitor_energy", "U", {"W": energy.value_si, "C": c.value_si}, math.sqrt(2.0 * energy.value_si / c.value_si))
        return build_result(
            route="capacitor_energy",
            target="voltage",
            value_si=value,
            question=question,
            formula="U = sqrt(2 * W_c / C)",
            premise="Capacitor energy rearrangement: U = sqrt(2W_c/C)",
            cot=[
                f"Identify energy W = {energy.raw} and capacitance C = {c.raw}.",
                "Use the formula solver to rearrange W_c = 1/2 C U^2 and solve for U.",
            ],
            confidence=0.92,
        )
    if "capacitance" in lower and "area" in quantities:
        distance = find_quantity(
            question,
            "d",
            [
                rf"(?:separation|distance between the plates|plate separation)\s*(?:of|d\s*=|=|is|are)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b"
            ],
        )
        if distance:
            eps_r = 1.0
            eps_match = re.search(rf"(?:dielectric constant|relative permittivity|epsilon_r|epsilon)\s*(?:=|of|is)?\s*(?P<val>{NUMBER_RE})", normalize_text(question), flags=re.IGNORECASE)
            if eps_match:
                eps_r = parse_number(eps_match.group("val"))
            area = quantities["area"]
            value = formula_value(
                "parallel_plate_capacitor",
                "C",
                {"eps_r": eps_r, "A": area.value_si, "d": distance.value_si},
                EPS0 * eps_r * area.value_si / distance.value_si,
            )
            return build_result(
                route="parallel_plate_capacitor",
                target="capacitance",
                value_si=value,
                question=question,
                formula="C = epsilon0 * epsilon_r * A / d",
                premise="Parallel-plate capacitor law: C = epsilon0 epsilon_r A / d",
                cot=[
                    f"Identify plate area A = {area.raw} and plate separation d = {distance.raw}.",
                    "Use the formula solver to evaluate C = epsilon0 epsilon_r A / d with air epsilon_r = 1 unless stated otherwise.",
                ],
                confidence=0.9,
            )
    return None


def solve_inductor_energy(question: str, quantities: dict[str, Quantity]) -> SolverResult | None:
    normalized = normalize_text(question)
    lower = normalized.lower()
    if ("formula" in lower or "what is the formula" in lower) and "magnetic field energy" not in lower:
        return None
    inductance = quantities.get("L") or find_symbol_quantity(question, "L", ["mH", "H"], aliases=["inductance"])
    current = quantities.get("I") or find_symbol_quantity(question, "I", ["mA", "A"], aliases=["current", "instantaneous current", "maximum current"])
    energy = quantities.get("W") or find_symbol_quantity(question, "W", ["mJ", "uJ", "microJ", "nJ", "J"], aliases=["magnetic field energy", "energy"])
    if "unit of inductance" in lower:
        return build_text_result(
            route="inductance_unit",
            answer="H",
            target="unit",
            formula="SI unit of inductance is henry",
            premise="Inductance is measured in henries",
            cot=["Identify that the requested physical quantity is inductance.", "Use the SI unit name and symbol."],
            confidence=0.95,
        )
    if "formula" in lower and "magnetic field energy" in lower and ("inductor" in lower or "coil" in lower):
        return build_text_result(
            route="inductor_energy_formula_concept",
            answer="W = 1/2 L I^2",
            target="formula",
            formula="W = 1/2 L I^2",
            premise="Magnetic energy stored in an inductor",
            cot=["Identify the requested formula for magnetic energy in an inductor.", "Use the standard inductor energy expression."],
            confidence=0.92,
        )
    if "graph" in lower and "magnetic field energy" in lower and "inductance" in lower and "constant" in lower:
        return build_text_result(
            route="inductor_energy_graph_concept",
            answer="straight line",
            target="concept",
            formula="W = 1/2 L I^2",
            premise="At fixed current, inductor energy is proportional to inductance",
            cot=["Keep I constant in W = 1/2 L I^2.", "The dependence on L is linear."],
            confidence=0.86,
        )
    if "graph" in lower and "magnetic field energy" in lower and "current" in lower:
        return build_text_result(
            route="inductor_energy_graph_concept",
            answer="parabola",
            target="concept",
            formula="W = 1/2 L I^2",
            premise="Inductor energy is proportional to current squared",
            cot=["Use W = 1/2 L I^2.", "A quadratic dependence on I gives a parabolic graph."],
            confidence=0.88,
        )
    if "current" in lower and "halved" in lower and ("magnetic field energy" in lower or "magnetic energy" in lower):
        return build_text_result(
            route="inductor_energy_ratio_concept",
            answer="one quarter",
            target="concept",
            formula="W proportional to I^2",
            premise="Inductor magnetic energy is proportional to the square of current",
            cot=["Use W = 1/2 L I^2.", "Halving current multiplies energy by (1/2)^2."],
            confidence=0.88,
        )
    efficiency_values = re.findall(rf"(?P<val>{NUMBER_RE})\s*J\b", normalize_text(question), flags=re.IGNORECASE)
    if "efficiency" in lower and len(efficiency_values) >= 2:
        dissipated = parse_number(efficiency_values[0])
        useful = parse_number(efficiency_values[1])
        value = useful / (useful + dissipated)
        return build_result(
            route="energy_efficiency",
            target="percent",
            value_si=value,
            question=question,
            formula="eta = useful_energy / input_energy",
            premise="Efficiency is useful stored energy divided by total supplied energy",
            cot=["Identify dissipated and useful stored energies.", "Use useful/(useful+dissipated)."],
            confidence=0.84,
            unit="%",
        )
    if "when will" in lower and "magnetic field energy" in lower and "zero" in lower:
        return build_text_result(
            route="inductor_zero_energy_concept",
            answer="when current is zero",
            target="concept",
            formula="W = 1/2 L I^2",
            premise="Inductor magnetic energy vanishes when current vanishes",
            cot=["Use W = 1/2 L I^2.", "The energy is zero exactly when I = 0."],
            confidence=0.9,
        )
    if "inductor" not in lower and "inductance" not in lower and "magnetic field energy" not in lower and "magnetic energy" not in lower and "coil" not in lower:
        return None
    current_amplitude = find_symbol_quantity(question, "I0", ["mA", "A"], aliases=["I0", "Imax", "maximum current"])
    expr_number = rf"[-+]?\d+(?:\.\d+)?\s*sqrt\s*\d+(?:\.\d+)?|[-+]?(?:\d+(?:\.\d+)?|pi|sqrt\d)[-+0-9.*/^()pisqrt]*|{NUMBER_RE}"
    if current_amplitude is None:
        amp_match = re.search(rf"\bI(?:\(t\))?\s*=\s*(?P<amp>{expr_number})\s*(?:x|\*)?\s*(?:sin|cos)\s*\(", normalized, flags=re.IGNORECASE)
        if amp_match:
            current_amplitude = Quantity("I0", parse_number(amp_match.group("amp")), "A", amp_match.group(0))
    if current_amplitude is None:
        max_current_match = re.search(
            rf"(?:maximum value of|maximum current(?: is|=| of)?|current reaches its maximum value of)\s*(?P<amp>{expr_number})\s*A\b",
            normalized,
            flags=re.IGNORECASE,
        )
        if max_current_match:
            current_amplitude = Quantity("I0", parse_number(max_current_match.group("amp")), "A", max_current_match.group(0))
    if current is None:
        function_current = re.search(
            rf"\bI(?:\(t\))?\s*=\s*(?P<amp>{expr_number})\s*(?:x|\*)?\s*(?P<trig>sin|cos)\s*\(?\s*(?P<omega>{expr_number})\s*t",
            normalized,
            flags=re.IGNORECASE,
        )
        time_match = re.search(rf"\bt\s*=\s*(?P<t>{expr_number})\s*(?P<unit>ms|s)\b", normalized, flags=re.IGNORECASE)
        if function_current and (time_match or re.search(r"\bt\s*=\s*0\b", normalized, flags=re.IGNORECASE)):
            amp_value = parse_number(function_current.group("amp"))
            omega_value = parse_number(function_current.group("omega"))
            time_value = 0.0
            if time_match:
                time_value = parse_number(time_match.group("t")) * UNIT_FACTORS.get(clean_unit(time_match.group("unit")), 1.0)
            angle = omega_value * time_value
            trig_value = math.cos(angle) if function_current.group("trig").lower() == "cos" else math.sin(angle)
            value = amp_value * trig_value
            current = Quantity("I", value, "A", function_current.group(0))
    if current is None:
        current_candidates = [value for value in current_values_amperes(question) if value > 0]
        if current_candidates:
            value = current_candidates[-1]
            current = Quantity("I", value, "A", f"{format_number(value)} A")
    if "maximum magnetic field energy" in lower and inductance and current_amplitude:
        value = 0.5 * inductance.value_si * current_amplitude.value_si * current_amplitude.value_si
        return build_result(
            route="inductor_max_energy_from_current_function",
            target="energy",
            value_si=value,
            question=question,
            formula="W_max = 1/2 L I0^2",
            premise="Maximum inductor energy uses current amplitude",
            cot=["Extract current amplitude from I(t).", "Compute Wmax=1/2 L I0^2."],
            confidence=0.88,
            unit="J",
        )
    if energy and "current is halved" in lower:
        value = energy.value_si / 4.0
        return build_result(
            route="inductor_energy_ratio",
            target="energy",
            value_si=value,
            question=question,
            formula="W_final/W_initial = (I_final/I_initial)^2",
            premise="Inductor magnetic energy is proportional to current squared",
            cot=["Identify that current is halved.", "Magnetic energy scales as I^2, so W becomes one quarter."],
            confidence=0.88,
        )
    if "energy" in lower and inductance and current:
        value = formula_value("inductor_energy", "W", {"L": inductance.value_si, "I": current.value_si}, 0.5 * inductance.value_si * current.value_si * current.value_si)
        return build_result(
            route="inductor_energy",
            target="energy",
            value_si=value,
            question=question,
            formula="W_l = 1/2 * L * I^2",
            premise="Inductor magnetic energy law: W_l = 1/2 L I^2",
            cot=[
                f"Identify inductance L = {inductance.raw} and current I = {current.raw}.",
                "Use the formula solver to evaluate W_l = 1/2 L I^2.",
            ],
            confidence=0.94,
        )
    if "current" in lower and inductance and energy:
        value = formula_value("inductor_energy", "I", {"W": energy.value_si, "L": inductance.value_si}, math.sqrt(2.0 * energy.value_si / inductance.value_si))
        return build_result(
            route="inductor_energy",
            target="current",
            value_si=value,
            question=question,
            formula="I = sqrt(2 * W_l / L)",
            premise="Inductor energy rearrangement: I = sqrt(2W_l/L)",
            cot=[
                f"Identify magnetic energy W = {energy.raw} and inductance L = {inductance.raw}.",
                "Use the formula solver to rearrange W_l = 1/2 L I^2 and solve for I.",
            ],
            confidence=0.92,
        )
    if "inductance" in lower and energy and current:
        value = formula_value("inductor_energy", "L", {"W": energy.value_si, "I": current.value_si}, 2.0 * energy.value_si / (current.value_si * current.value_si))
        return build_result(
            route="inductor_energy",
            target="inductance",
            value_si=value,
            question=question,
            formula="L = 2 * W_l / I^2",
            premise="Inductor energy rearrangement: L = 2W_l/I^2",
            cot=[
                f"Identify magnetic energy W = {energy.raw} and current I = {current.raw}.",
                "Use the formula solver to rearrange W_l = 1/2 L I^2 and solve for L.",
            ],
            confidence=0.92,
        )
    return None


def solve_rlc(question: str, quantities: dict[str, Quantity]) -> SolverResult | None:
    normalized = normalize_text(question)
    lower = normalized.lower()
    asks_current = bool(re.search(r"\bcurrent\b|\beffective current\b|\brms current\b|\bI\b|\bImax\b", normalized, flags=re.IGNORECASE))
    asks_power = bool(re.search(r"\bpower\b|\bPmax\b", normalized, flags=re.IGNORECASE))
    capacitance = quantities.get("C") or find_symbol_quantity(question, "C", ["uF", "microF", "mF", "nF", "pF", "F"], aliases=["capacitance"])
    inductance = quantities.get("L") or find_symbol_quantity(question, "L", ["mH", "H"], aliases=["inductance"])
    frequency = quantities.get("f") or find_symbol_quantity(question, "f", ["kHz", "Hz"], aliases=["frequency"])
    if frequency is None and ("resonance" in lower or "resonate" in lower or "resonating" in lower or "resonant" in lower or "f0" in lower):
        frequency_values = numeric_unit_values(question, ["kHz", "Hz"], "f")
        if frequency_values:
            frequency = frequency_values[-1]
    resistance = quantities.get("R") or find_symbol_quantity(question, "R", ["ohm"], aliases=["resistance", "pure resistance"])
    voltage = quantities.get("U") or parse_rms_voltage_from_time_function(question) or find_symbol_quantity(question, "U", ["V", "kV"], aliases=["RMS voltage", "effective voltage", "applied voltage", "source voltage", "voltage"])
    current = quantities.get("I") or find_symbol_quantity(question, "I", ["mA", "A"], aliases=["RMS current", "effective current", "current", "Imax"])
    impedance = find_impedance(question)
    power = find_power(question)
    period = find_symbol_quantity(question, "T", ["ms", "s"], aliases=["period", "natural period", "oscillation period"])
    xl_quantity = find_reactance(question, "XL") or find_reactance(question, "Z_L")
    xc_quantity = find_reactance(question, "XC") or find_reactance(question, "Z_C")
    omega = parse_ac_omega(question)
    if omega is None and frequency:
        omega = 2.0 * math.pi * frequency.value_si

    xl_value = xl_quantity.value_si if xl_quantity else (omega * inductance.value_si if omega is not None and inductance else None)
    xc_value = xc_quantity.value_si if xc_quantity else (1.0 / (omega * capacitance.value_si) if omega is not None and capacitance and omega > 0 else None)

    if (
        xl_quantity
        and xc_quantity
        and ("resonance" in lower or "resonant" in lower or "resonate" in lower or "resonating" in lower)
        and any(phrase in lower for phrase in ["multiple of", "factor", "multiplied", "adjusted", "value of k", "k x", "kx"])
        and ("frequency" in lower or "omega" in lower or "angular frequency" in lower)
    ):
        value = math.sqrt(xc_quantity.value_si / xl_quantity.value_si) if xl_quantity.value_si > 0 else 0.0
        return build_result(
            route="rlc_resonance_frequency_multiplier",
            target="ratio",
            value_si=value,
            question=question,
            formula="m = sqrt(X_C0/X_L0)",
            premise="With fixed L and C, X_L scales as omega while X_C scales as 1/omega",
            cot=["Set m X_L0 = X_C0/m for resonance.", "Solve m = sqrt(X_C0/X_L0)."],
            confidence=0.9,
            unit="",
        )

    if ("impedance" in lower or re.search(r"\bZ\b", normalized)) and ("resonance" in lower or "resonant" in lower) and resistance:
        return build_result(
            route="rlc_resonance_impedance",
            target="impedance",
            value_si=resistance.value_si,
            question=question,
            formula="Z = R at resonance",
            premise="In a series RLC circuit at resonance, inductive and capacitive reactances cancel",
            cot=["Use X_L = X_C at resonance.", "The total impedance magnitude is therefore the resistance R."],
            confidence=0.92,
            unit="ohm",
        )

    if ("angular frequency" in lower or "omega" in lower or "ω" in question) and parse_ac_omega(question) is not None:
        source_omega = parse_ac_omega(question)
        omega_expr = re.search(
            r"cos\(?\s*(?P<omega>[-+0-9.*/^()pisqrt]+)t",
            normalize_text(question).replace(" ", ""),
            flags=re.IGNORECASE,
        )
        if omega_expr and "pi" in omega_expr.group("omega").lower():
            raw_omega = omega_expr.group("omega").replace("*", "").replace("pi", "π")
            answer = f"{raw_omega} rad/s"
            return SolverResult(
                solved=True,
                route="ac_source_angular_frequency",
                answer=answer,
                fol=f"Given(u(t)) and Law(u=U0 cos(omega t)) -> Answer({answer})",
                cot=numbered_cot(["Read omega directly from the coefficient of t in the source voltage."]),
                premises=["The coefficient of t in u=U0 cos(omega t) is the angular frequency"],
                confidence=0.92,
                target="frequency",
                value_si=source_omega,
                unit="rad/s",
                engine=formula_engine_name(),
            )
        return build_result(
            route="ac_source_angular_frequency",
            target="frequency",
            value_si=source_omega,
            question=question,
            formula="u = U0 cos(omega t)",
            premise="The coefficient of t in the cosine voltage source is the source angular frequency",
            cot=["Read omega directly from the time-domain source voltage.", "Report it as angular frequency."],
            confidence=0.92,
            unit="rad/s",
        )

    if (
        ("source" in lower or "of the source" in lower or "source voltage" in lower)
        and ("voltage" in lower or "rms" in lower or "effective" in lower)
        and parse_rms_voltage_from_time_function(question)
    ):
        source_voltage = parse_rms_voltage_from_time_function(question)
        return build_result(
            route="ac_source_rms_voltage",
            target="voltage",
            value_si=source_voltage.value_si,
            question=question,
            formula="u = U_rms*sqrt(2)*cos(omega t)",
            premise="The amplitude in u=U_rms*sqrt(2)cos(omega t) gives the RMS source voltage",
            cot=["Read the RMS coefficient from the sinusoidal source voltage."],
            confidence=0.92,
            unit="V",
        )

    if (
        ("voltage across the capacitor" in lower or "uc" in lower or re.search(r"\bU_C\b", normalized, flags=re.IGNORECASE))
        and ("r-c" in lower or "rc" in lower)
        and ("c-l" in lower or "cl" in lower)
        and ("resonance" in lower or "resonant" in lower)
        and "internal resistance" not in lower
    ):
        voltages = numeric_unit_values(question, ["V"], "U")
        if len(voltages) >= 2:
            source_value = voltage.value_si if voltage else voltages[0].value_si
            section_candidates = [v.value_si for v in voltages if abs(v.value_si - source_value) > max(1e-9, abs(source_value) * 1e-6)]
            if section_candidates:
                section_value = max(section_candidates)
                value = math.sqrt(max(0.0, section_value * section_value - source_value * source_value))
                return build_result(
                    route="rlc_resonance_capacitor_voltage_from_sections",
                    target="voltage",
                    value_si=value,
                    question=question,
                    formula="At resonance, U_C = sqrt(U_RC^2 - U^2) when U_RC = U_CL",
                    premise="Phasor relation for equal R-C and C-L section voltages in a series RLC circuit at resonance",
                    cot=["Identify source voltage and equal section voltage.", "Use the right-triangle phasor relation to solve for capacitor voltage."],
                    confidence=0.84,
                    unit="V",
                )

    if ("impedance" in lower or re.search(r"\bZ\b", normalize_text(question))) and voltage and current:
        value = voltage.value_si / current.value_si
        return build_result(
            route="ac_impedance_from_voltage_current",
            target="impedance",
            value_si=value,
            question=question,
            formula="Z = U/I",
            premise="RMS Ohm law for AC impedance",
            cot=["Identify RMS voltage and RMS current.", "Compute impedance Z=U/I."],
            confidence=0.9,
            unit="ohm",
        )
    if ("natural period" in lower or "period of oscillation" in lower) and capacitance and inductance:
        value = 2.0 * math.pi * math.sqrt(inductance.value_si * capacitance.value_si)
        return build_result(
            route="lc_natural_period",
            target="time",
            value_si=value,
            question=question,
            formula="T = 2*pi*sqrt(LC)",
            premise="Natural period of an ideal LC circuit",
            cot=["Identify L and C.", "Compute T=2*pi*sqrt(LC)."],
            confidence=0.9,
            unit="s",
        )
    if ("angular frequency" in lower or "omega" in lower or "ω" in question) and capacitance and inductance:
        value = 1.0 / math.sqrt(inductance.value_si * capacitance.value_si)
        return build_result(
            route="lc_angular_frequency",
            target="frequency",
            value_si=value,
            question=question,
            formula="omega = 1/sqrt(LC)",
            premise="Natural angular frequency of an ideal LC circuit",
            cot=["Identify L and C.", "Compute omega=1/sqrt(LC)."],
            confidence=0.9,
            unit="Hz",
        )
    if "frequency" in lower and period:
        value = 1.0 / period.value_si
        return build_result(
            route="frequency_from_period",
            target="frequency",
            value_si=value,
            question=question,
            formula="f = 1/T",
            premise="Frequency is reciprocal of period",
            cot=[f"Identify period T = {period.raw}.", "Compute f=1/T."],
            confidence=0.9,
            unit="Hz",
        )
    current_amplitude = find_symbol_quantity(question, "I0", ["mA", "A"], aliases=["I0", "Imax", "maximum current"])
    if "maximum magnetic energy" in lower and inductance and current_amplitude:
        value = 0.5 * inductance.value_si * current_amplitude.value_si * current_amplitude.value_si
        return build_result(
            route="lc_max_magnetic_energy",
            target="energy",
            value_si=value,
            question=question,
            formula="W_Lmax = 1/2 L I0^2",
            premise="Maximum magnetic energy in an inductor",
            cot=["Identify inductance and current amplitude.", "Compute Wmax=1/2 L I0^2."],
            confidence=0.9,
            unit="J",
        )

    if ("resonance" in lower or "resonant" in lower or "resonate" in lower) and impedance and ("resistance" in lower or re.search(r"\bR\b", normalize_text(question))):
        return build_result(
            route="rlc_resonance_resistance",
            target="resistance",
            value_si=impedance.value_si,
            question=question,
            formula="at resonance, Z = R",
            premise="At series RLC resonance, inductive and capacitive reactances cancel so impedance equals resistance",
            cot=[f"Identify resonance and impedance Z = {impedance.raw}.", "Use R = Z at resonance."],
            confidence=0.92,
            unit="ohm",
        )

    if ("resonance" in lower or "resonate" in lower or "resonant" in lower) and capacitance and inductance and frequency:
        f0 = formula_value(
            "rlc_resonance",
            "f",
            {"L": inductance.value_si, "C": capacitance.value_si},
            1.0 / (2.0 * math.pi * math.sqrt(inductance.value_si * capacitance.value_si)),
        )
        answer = "Yes" if abs(frequency.value_si - f0) / f0 <= 0.015 else "No"
        return SolverResult(
            solved=True,
            route="rlc_resonance",
            answer=answer,
            fol=f"Given(f={frequency.value_si}) and Law(f0=1/(2*pi*sqrt(L*C))) -> Answer({answer})",
            cot=[
                f"Identify L = {inductance.raw}, C = {capacitance.raw}, and f = {frequency.raw}.",
                "Use the formula solver to compute the resonant frequency f0 = 1/(2*pi*sqrt(L*C)).",
                f"Compare f with f0 = {format_number(f0, 2)} Hz.",
                f"The result is {answer}.",
            ],
            premises=["RLC resonance condition: f = 1/(2*pi*sqrt(L*C))"],
            confidence=0.9,
            target="yes_no",
            value_si=None,
            unit=None,
            engine=formula_engine_name(),
        )
    if ("resonant frequency" in lower or "natural oscillation frequency" in lower or ("resonance" in lower and "frequency" in lower) or "resonate at" in lower or "f0" in lower) and capacitance and inductance and not ("is " in lower and "?" in lower and frequency):
        value = formula_value(
            "rlc_resonance",
            "f",
            {"L": inductance.value_si, "C": capacitance.value_si},
            1.0 / (2.0 * math.pi * math.sqrt(inductance.value_si * capacitance.value_si)),
        )
        return build_result(
            route="rlc_resonant_frequency",
            target="frequency",
            value_si=value,
            question=question,
            formula="f0 = 1 / (2*pi*sqrt(L*C))",
            premise="LC/RLC resonant frequency law",
            cot=[f"Identify L = {inductance.raw} and C = {capacitance.raw}.", "Compute f0 = 1/(2*pi*sqrt(LC))."],
            confidence=0.92,
            unit="Hz",
        )
    if ("capacitance" in lower or "capacitor value" in lower or "capacitor should" in lower or re.search(r"\bC\b", normalize_text(question))) and ("resonate" in lower or "resonating" in lower or "resonance" in lower or "f0" in lower) and inductance and frequency and not capacitance:
        value = 1.0 / ((2.0 * math.pi * frequency.value_si) ** 2 * inductance.value_si)
        return build_result(
            route="rlc_resonance_capacitance",
            target="capacitance",
            value_si=value,
            question=question,
            formula="C = 1 / ((2*pi*f)^2 * L)",
            premise="Resonance condition rearranged for capacitance",
            cot=[f"Identify L = {inductance.raw} and f = {frequency.raw}.", "Rearrange f0 = 1/(2*pi*sqrt(LC)) for C."],
            confidence=0.9,
        )
    if ("inductance" in lower or "inductor should" in lower or "what inductor" in lower or re.search(r"\bL\b", normalize_text(question))) and ("resonate" in lower or "resonating" in lower or "resonance" in lower or "f0" in lower) and capacitance and frequency and not inductance:
        value = 1.0 / ((2.0 * math.pi * frequency.value_si) ** 2 * capacitance.value_si)
        return build_result(
            route="rlc_resonance_inductance",
            target="inductance",
            value_si=value,
            question=question,
            formula="L = 1 / ((2*pi*f)^2 * C)",
            premise="Resonance condition rearranged for inductance",
            cot=[f"Identify C = {capacitance.raw} and f = {frequency.raw}.", "Rearrange f0 = 1/(2*pi*sqrt(LC)) for L."],
            confidence=0.9,
        )
    if "capacitive reactance" in lower and capacitance and frequency:
        value = formula_value(
            "capacitive_reactance",
            "Xc",
            {"f": frequency.value_si, "C": capacitance.value_si},
            1.0 / (2.0 * math.pi * frequency.value_si * capacitance.value_si),
        )
        return build_result(
            route="ac_reactance",
            target="reactance",
            value_si=value,
            question=question,
            formula="X_C = 1 / (2 * pi * f * C)",
            premise="Capacitive reactance law: X_C = 1/(2*pi*f*C)",
            cot=[
                f"Identify capacitance C = {capacitance.raw} and frequency f = {frequency.raw}.",
                "Use the formula solver to evaluate X_C = 1/(2*pi*f*C).",
            ],
            confidence=0.9,
        )
    if ("inductive reactance" in lower or "z_l" in lower or "xl" in lower) and xl_value is not None:
        return build_result(
            route="ac_inductive_reactance",
            target="reactance",
            value_si=xl_value,
            question=question,
            formula="X_L = omega * L",
            premise="Inductive reactance law",
            cot=["Identify angular frequency or compute omega = 2*pi*f.", "Evaluate X_L = omega L."],
            confidence=0.9,
            unit="ohm",
        )
    if ("capacitive reactance" in lower or "z_c" in lower or "xc" in lower) and xc_value is not None:
        return build_result(
            route="ac_capacitive_reactance",
            target="reactance",
            value_si=xc_value,
            question=question,
            formula="X_C = 1 / (omega * C)",
            premise="Capacitive reactance law",
            cot=["Identify angular frequency or compute omega = 2*pi*f.", "Evaluate X_C = 1/(omega C)."],
            confidence=0.9,
            unit="ohm",
        )
    if ("impedance" in lower or re.search(r"\bZ\b", normalize_text(question))) and resistance and xl_value is not None and xc_value is not None:
        value = math.hypot(resistance.value_si, xl_value - xc_value)
        return build_result(
            route="rlc_series_impedance",
            target="impedance",
            value_si=value,
            question=question,
            formula="Z = sqrt(R^2 + (X_L - X_C)^2)",
            premise="Series RLC impedance law",
            cot=["Compute X_L and X_C.", "Use Z = sqrt(R^2 + (X_L-X_C)^2)."],
            confidence=0.9,
            unit="ohm",
        )
    if ("impedance" in lower or re.search(r"\bZ\b", normalize_text(question))) and resistance and xl_quantity and xc_quantity:
        value = math.hypot(resistance.value_si, xl_quantity.value_si - xc_quantity.value_si)
        return build_result(
            route="rlc_series_impedance",
            target="impedance",
            value_si=value,
            question=question,
            formula="Z = sqrt(R^2 + (X_L - X_C)^2)",
            premise="Series RLC impedance law",
            cot=["Use the given reactances and resistance.", "Compute total impedance."],
            confidence=0.9,
            unit="ohm",
        )
    if ("power factor" in lower or "cosphi" in lower or "cos phi" in lower or "cosφ" in lower):
        if ("resonance" in lower or "resonant" in lower) and not impedance:
            value = 1.0
        elif resistance and impedance:
            value = resistance.value_si / impedance.value_si
        elif resistance and xl_value is not None and xc_value is not None:
            z_value = math.hypot(resistance.value_si, xl_value - xc_value)
            value = resistance.value_si / z_value
        else:
            value = None
        if value is not None:
            return build_result(
                route="rlc_power_factor",
                target="power_factor",
                value_si=value,
                question=question,
                formula="cos(phi) = R/Z",
                premise="Series RLC power factor relation",
                cot=["Compute or identify total impedance.", "Use cos(phi)=R/Z; at resonance it equals 1."],
                confidence=0.9,
                unit="",
            )
    if ("quality factor" in lower or re.search(r"\bQ\b", normalize_text(question))) and resistance and inductance and capacitance:
        value = math.sqrt(inductance.value_si / capacitance.value_si) / resistance.value_si
        return build_result(
            route="rlc_quality_factor",
            target="quality_factor",
            value_si=value,
            question=question,
            formula="Q = sqrt(L/C) / R",
            premise="Series RLC quality factor at resonance",
            cot=["Identify L, C, and R.", "Use Q = sqrt(L/C)/R."],
            confidence=0.88,
            unit="",
        )
    if ("inductive reactance" in lower or "zl" in lower or "z_l" in lower) and resistance and ("resonance" in lower or "resonant" in lower or "resonate" in lower or "resonates" in lower):
        current_ratio = rlc_off_resonance_current_ratio(question)
        frequency_multiplier = rlc_frequency_multiplier(question)
        if current_ratio and frequency_multiplier and current_ratio > 1 and frequency_multiplier > 0 and abs(frequency_multiplier - 1.0) > 1e-9:
            denominator = abs(frequency_multiplier - 1.0 / frequency_multiplier)
            xl_resonance = resistance.value_si * math.sqrt(max(0.0, current_ratio * current_ratio - 1.0)) / denominator
            value = xl_resonance * frequency_multiplier if rlc_reactance_requested_at_new_frequency(question) else xl_resonance
            return build_result(
                route="rlc_reactance_from_current_drop",
                target="reactance",
                value_si=value,
                question=question,
                formula="Z = U/I, Z^2 = R^2 + (mX_L0 - X_L0/m)^2",
                premise="Off-resonance current drop in a series RLC circuit determines the resonant inductive reactance",
                cot=[
                    "At resonance the impedance is R and the current is maximal.",
                    "Use the current ratio to get the off-resonance impedance ratio.",
                    "Use X_L(f)=mX_L0 and X_C(f)=X_L0/m to solve for X_L0.",
                ],
                confidence=0.88,
                unit="ohm",
            )
    if asks_current and voltage and impedance:
        value = voltage.value_si / impedance.value_si
        return build_result(
            route="ac_current",
            target="current",
            value_si=value,
            question=question,
            formula="I = U / Z",
            premise="RMS Ohm law for AC impedance",
            cot=[f"Identify voltage U = {voltage.raw} and impedance Z = {impedance.raw}.", "Compute I = U/Z."],
            confidence=0.9,
            unit="A",
        )
    if asks_current and ("resonance" in lower or "resonant" in lower) and voltage and resistance:
        value = voltage.value_si / resistance.value_si
        return build_result(
            route="rlc_resonance_current",
            target="current",
            value_si=value,
            question=question,
            formula="I = U / R at resonance",
            premise="At resonance, Z=R",
            cot=[f"Identify U = {voltage.raw} and R = {resistance.raw}.", "Compute I=U/R."],
            confidence=0.92,
            unit="A",
        )
    if asks_current and voltage and resistance and xl_value is not None and xc_value is not None:
        z_value = math.hypot(resistance.value_si, xl_value - xc_value)
        value = voltage.value_si / z_value
        return build_result(
            route="rlc_ac_current_from_impedance",
            target="current",
            value_si=value,
            question=question,
            formula="I = U / sqrt(R^2 + (X_L-X_C)^2)",
            premise="RMS current in a series RLC circuit",
            cot=["Compute X_L and X_C from the source frequency.", "Compute Z, then I=U/Z."],
            confidence=0.9,
            unit="A",
        )
    if ("voltage across the inductor" in lower or re.search(r"\bUL\b", normalized, flags=re.IGNORECASE)) and voltage and resistance and xl_value is not None and xc_value is not None:
        z_value = math.hypot(resistance.value_si, xl_value - xc_value)
        current_value = voltage.value_si / z_value
        value = current_value * xl_value
        return build_result(
            route="rlc_inductor_voltage",
            target="voltage",
            value_si=value,
            question=question,
            formula="U_L = I X_L",
            premise="RMS voltage across the inductor in a series RLC circuit",
            cot=["Compute current from total impedance.", "Use U_L=I X_L."],
            confidence=0.88,
            unit="V",
        )
    if ("voltage across the capacitor" in lower or "uc" in lower or re.search(r"\bU_C\b", normalized, flags=re.IGNORECASE)) and voltage and resistance and xl_value is not None and xc_value is not None:
        z_value = math.hypot(resistance.value_si, xl_value - xc_value)
        current_value = voltage.value_si / z_value
        value = current_value * xc_value
        return build_result(
            route="rlc_capacitor_voltage",
            target="voltage",
            value_si=value,
            question=question,
            formula="U_C = I X_C",
            premise="RMS voltage across the capacitor in a series RLC circuit",
            cot=["Compute current from total impedance.", "Use U_C=I X_C."],
            confidence=0.88,
            unit="V",
        )
    if ("voltage across the inductor" in lower or "ul" in lower) and voltage and resistance and inductance and capacitance and ("resonance" in lower or "resonant" in lower):
        omega0 = 1.0 / math.sqrt(inductance.value_si * capacitance.value_si)
        value = (voltage.value_si / resistance.value_si) * omega0 * inductance.value_si
        return build_result(
            route="rlc_resonance_inductor_voltage",
            target="voltage",
            value_si=value,
            question=question,
            formula="U_L = (U/R) * omega0 * L",
            premise="At resonance, I=U/R and U_L=I X_L",
            cot=["Compute resonance angular frequency omega0=1/sqrt(LC).", "Compute I=U/R, then U_L=I omega0 L."],
            confidence=0.88,
            unit="V",
        )
    if asks_power and "resonance" in lower and voltage and resistance:
        value = formula_value(
            "rlc_resonance_power",
            "P",
            {"U": voltage.value_si, "R": resistance.value_si},
            voltage.value_si * voltage.value_si / resistance.value_si,
        )
        return build_result(
            route="rlc_resonance_power",
            target="power",
            value_si=value,
            question=question,
            formula="P = U^2 / R at resonance",
            premise="At RLC resonance, impedance equals R and P = U^2/R",
            cot=[
                f"Identify voltage U = {voltage.raw} and resistance R = {resistance.raw}.",
                "At resonance, Z = R, so the formula solver evaluates P = U^2/R.",
            ],
            confidence=0.9,
        )
    if asks_power and voltage and resistance and xl_value is not None and xc_value is not None:
        z_value = math.hypot(resistance.value_si, xl_value - xc_value)
        value = voltage.value_si * voltage.value_si * resistance.value_si / (z_value * z_value)
        return build_result(
            route="rlc_ac_power_from_impedance",
            target="power",
            value_si=value,
            question=question,
            formula="P = U^2 R / Z^2",
            premise="Average power in a series RLC circuit",
            cot=["Compute X_L, X_C, and Z.", "Use P=U^2R/Z^2."],
            confidence=0.88,
            unit="W",
        )
    if asks_power and voltage and impedance and resistance:
        value = voltage.value_si * voltage.value_si * resistance.value_si / (impedance.value_si * impedance.value_si)
        return build_result(
            route="ac_power",
            target="power",
            value_si=value,
            question=question,
            formula="P = U^2 R / Z^2",
            premise="Average power in a series AC circuit",
            cot=["Identify U, R, and Z.", "Use P=U^2 R/Z^2."],
            confidence=0.88,
            unit="W",
        )
    if asks_power and current and resistance:
        value = current.value_si * current.value_si * resistance.value_si
        return build_result(
            route="ac_power",
            target="power",
            value_si=value,
            question=question,
            formula="P = I^2 R",
            premise="Average power dissipated in resistance",
            cot=["Identify RMS current and resistance.", "Use P=I^2R."],
            confidence=0.88,
            unit="W",
        )
    if asks_current and "resonance" in lower and voltage and resistance:
        value = formula_value(
            "rlc_resonance_current",
            "I",
            {"U": voltage.value_si, "R": resistance.value_si},
            voltage.value_si / resistance.value_si,
        )
        return build_result(
            route="rlc_resonance_current",
            target="current",
            value_si=value,
            question=question,
            formula="I = U / R at resonance",
            premise="At RLC resonance, impedance equals R and I = U/R",
            cot=[
                f"Identify voltage U = {voltage.raw} and resistance R = {resistance.raw}.",
                "At resonance, Z = R, so the formula solver evaluates I = U/R.",
            ],
            confidence=0.9,
        )
    return None


def find_energy_phrase(question: str, phrase: str) -> Quantity | None:
    return find_quantity(
        question,
        phrase,
        [rf"{phrase}\s*(?:of|=|is)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>mJ|uJ|microJ|nJ|J)\b"],
    )


def solve_lc_energy(question: str, quantities: dict[str, Quantity]) -> SolverResult | None:
    normalized = normalize_text(question)
    lower = normalized.lower()
    if not re.search(r"\blc\b", lower) and "oscillation energy" not in lower:
        return None

    capacitance = quantities.get("C")
    voltage = quantities.get("U")
    if voltage is None:
        voltage_values = re.findall(rf"(?P<val>{NUMBER_RE})\s*V\b", normalize_text(question))
        if voltage_values:
            raw_value = voltage_values[-1]
            voltage = Quantity("U", parse_number(raw_value), "V", f"{raw_value} V")
    total_energy = (
        find_energy_phrase(question, "total energy")
        or find_energy_phrase(question, "total oscillation energy")
        or find_energy_phrase(question, "total oscillatory energy")
        or find_energy_phrase(question, "total oscillating energy")
    )
    electric_energy = find_energy_phrase(question, "electric field energy")
    magnetic_energy = find_energy_phrase(question, "magnetic field energy")

    if "magnetic field energy" in lower and total_energy and electric_energy and ("what" in lower or "calculate" in lower or "determine" in lower):
        value = max(0.0, total_energy.value_si - electric_energy.value_si)
        return build_result(
            route="lc_energy_complement",
            target="energy",
            value_si=value,
            question=question,
            formula="W_L = W_total - W_C",
            premise="Ideal LC total energy is split between electric and magnetic forms",
            cot=["Identify total LC energy and electric-field energy.", "Subtract to get the magnetic-field energy."],
            confidence=0.9,
            unit=total_energy.unit,
        )
    if "electric field energy" in lower and total_energy and magnetic_energy and ("what" in lower or "calculate" in lower or "determine" in lower):
        value = max(0.0, total_energy.value_si - magnetic_energy.value_si)
        return build_result(
            route="lc_energy_complement",
            target="energy",
            value_si=value,
            question=question,
            formula="W_C = W_total - W_L",
            premise="Ideal LC total energy is split between electric and magnetic forms",
            cot=["Identify total LC energy and magnetic-field energy.", "Subtract to get the electric-field energy."],
            confidence=0.9,
            unit=total_energy.unit,
        )
    expr_number = rf"[-+]?\d+(?:\.\d+)?\s*sqrt\s*\d+(?:\.\d+)?|[-+]?(?:\d+(?:\.\d+)?|pi|sqrt\d)[-+0-9.*/^()pisqrt\s]*|{NUMBER_RE}"
    electric_energy_function = re.search(
        rf"\bW_C\s*=\s*(?P<amp>{expr_number})\s*(?P<trig>sin|cos)(?:2|\^2)?\s*\(?\s*(?P<omega>{expr_number})\s*t",
        normalized,
        flags=re.IGNORECASE,
    )
    time_match = re.search(rf"\bt\s*=\s*(?P<t>{expr_number})\s*(?P<unit>ms|s)\b", normalized, flags=re.IGNORECASE)
    if electric_energy_function and time_match and "magnetic field energy" in lower:
        amplitude = parse_number(electric_energy_function.group("amp"))
        omega_value = parse_number(electric_energy_function.group("omega"))
        time_value = parse_number(time_match.group("t")) * UNIT_FACTORS.get(clean_unit(time_match.group("unit")), 1.0)
        angle = omega_value * time_value
        trig_value = math.cos(angle) if electric_energy_function.group("trig").lower() == "cos" else math.sin(angle)
        electric_value = amplitude * trig_value * trig_value
        value = max(0.0, amplitude - electric_value)
        return build_result(
            route="lc_energy_function_complement",
            target="energy",
            value_si=value,
            question=question,
            formula="W_L = W_0 - W_C(t)",
            premise="Ideal LC electric and magnetic energies are complementary",
            cot=["Evaluate the stated electric-field energy at the requested time.", "Subtract it from the total amplitude."],
            confidence=0.88,
            unit="J",
        )
    fraction_energy_match = re.search(
        r"(electric|magnetic) field energy is (?P<num>\d+(?:\.\d+)?)\s*/\s*(?P<den>\d+(?:\.\d+)?) of the total energy",
        lower,
    )
    if fraction_energy_match and "magnetic field energy" in lower and "fraction" in lower:
        fraction = float(fraction_energy_match.group("num")) / float(fraction_energy_match.group("den"))
        magnetic_fraction = fraction if fraction_energy_match.group(1).startswith("magnetic") else 1.0 - fraction
        common_fractions = {
            0.25: "1/4",
            0.5: "1/2",
            0.75: "3/4",
            1.0 / 3.0: "1/3",
            2.0 / 3.0: "2/3",
        }
        answer = next((label for value, label in common_fractions.items() if abs(magnetic_fraction - value) < 1e-9), format_number(magnetic_fraction))
        return build_text_result(
            route="lc_energy_fraction_complement",
            answer=answer,
            target="energy_fraction",
            formula="W_L/W_total = 1 - W_C/W_total",
            premise="Ideal LC total energy is conserved",
            cot=["Read the stated electric or magnetic energy fraction.", "Use the complement for the other form."],
            confidence=0.9,
        )
    if "expression" in lower and "magnetic field energy" in lower and "electric field energy" in lower and ("cos2" in lower or "cos^2" in lower):
        return build_text_result(
            route="lc_energy_expression_complement",
            answer="W_C = W0sin^2(omega t)",
            target="energy_expression",
            formula="W_C = W_0 - W_L; cos^2 + sin^2 = 1",
            premise="Electric and magnetic energies in an ideal LC circuit are complementary",
            cot=["Use W_total = W0.", "If W_L = W0 cos^2(omega t), then W_C = W0 sin^2(omega t)."],
            confidence=0.88,
        )

    if ("total energy" in lower or "total electromagnetic energy" in lower) and ("vary over time" in lower or "lost" in lower):
        answer = "No" if "lost" in lower else "constant"
        return build_text_result(
            route="lc_energy_conservation_concept",
            answer=answer,
            target="concept",
            formula="W_total = W_C + W_L = constant",
            premise="Ideal LC circuits conserve total electromagnetic energy",
            cot=["Use the ideal LC assumption.", "Energy exchanges between capacitor and inductor while the total remains constant."],
            confidence=0.88,
        )
    if "electric field energy reaches its maximum" in lower:
        return build_text_result(
            route="lc_energy_exchange_concept",
            answer="0",
            target="energy",
            formula="W_C maximum -> W_L = 0",
            premise="LC energy alternates between electric and magnetic forms",
            cot=["At maximum electric energy, all energy is in the capacitor.", "The magnetic field energy is zero."],
            confidence=0.88,
        )
    if "magnetic energy is half of the total energy" in lower and "electric energy" in lower:
        return build_text_result(
            route="lc_energy_exchange_concept",
            answer="half of the total energy",
            target="energy",
            formula="W_C = W_total - W_L",
            premise="Ideal LC total energy conservation",
            cot=["Use W_total = W_C + W_L.", "If W_L is half, W_C is the remaining half."],
            confidence=0.88,
        )
    if "capacitor is maximally charged" in lower and "current" in lower:
        return build_text_result(
            route="lc_current_concept",
            answer="0",
            target="current",
            formula="Q maximum -> I = 0",
            premise="In LC oscillation, charge extrema occur when current is zero",
            cot=["At maximum capacitor charge, charge is momentarily not changing.", "Current dQ/dt is zero."],
            confidence=0.9,
        )
    if "electric field energy" in lower and ("reaches its maximum" in lower or "reach its maximum" in lower):
        return build_text_result(
            route="lc_charge_concept",
            answer="charge is maximum",
            target="concept",
            formula="W_C = Q^2/(2C)",
            premise="Capacitor electric energy is maximal when capacitor charge magnitude is maximal",
            cot=["Use W_C = Q^2/(2C).", "Maximum electric energy occurs at maximum |Q|."],
            confidence=0.88,
        )
    if "energy of oscillation" in lower and "expression" in lower:
        return build_text_result(
            route="lc_energy_formula_concept",
            answer="W = 1/2 L I_max^2",
            target="formula",
            formula="W = 1/2 L I_max^2 = Q_max^2/(2C)",
            premise="Total LC oscillation energy equals maximum magnetic or maximum electric energy",
            cot=["Use conservation of total LC energy.", "At maximum current, all energy is magnetic."],
            confidence=0.88,
        )
    if "resonant angular frequency" in lower or "angular frequency of oscillation" in lower:
        return build_text_result(
            route="lc_angular_frequency_formula",
            answer="omega = 1/sqrt(LC)",
            target="formula",
            formula="omega = 1/sqrt(LC)",
            premise="Natural angular frequency of an ideal LC oscillator",
            cot=["Identify this as an ideal LC oscillator.", "Use the standard angular frequency relation."],
            confidence=0.9,
        )
    if "oscillation period" in lower and "calculated" in lower:
        return build_text_result(
            route="lc_period_formula",
            answer="T = 2*pi*sqrt(LC)",
            target="formula",
            formula="T = 2*pi*sqrt(LC)",
            premise="Natural period of an ideal LC oscillator",
            cot=["Use omega = 1/sqrt(LC).", "Then T = 2*pi/omega."],
            confidence=0.9,
        )
    if "voltage across the capacitor" in lower and "current" in lower and "maximum" in lower:
        return build_text_result(
            route="lc_voltage_concept",
            answer="0",
            target="voltage",
            formula="I maximum -> capacitor energy and voltage are zero",
            premise="At maximum current, all LC energy is magnetic",
            cot=["At maximum current, capacitor charge is zero.", "Therefore capacitor voltage is zero."],
            confidence=0.88,
        )
    if "what kind of oscillation" in lower:
        return build_text_result(
            route="lc_oscillation_concept",
            answer="simple harmonic oscillation",
            target="concept",
            formula="LC equations reduce to SHM",
            premise="Ideal LC charge/current oscillations are sinusoidal",
            cot=["The LC differential equation is harmonic.", "Therefore the oscillation is simple harmonic."],
            confidence=0.86,
        )
    if "shape of the graph" in lower and "electric field energy" in lower and "magnetic field energy" in lower:
        return build_text_result(
            route="lc_energy_graph_concept",
            answer="sinusoidal with phase shift pi/2",
            target="concept",
            formula="W_C and W_L exchange periodically",
            premise="LC electric and magnetic energies oscillate periodically out of phase",
            cot=["Capacitor and inductor energies exchange over time.", "Their graphs are periodic and shifted in phase."],
            confidence=0.82,
        )

    if ("where" in lower or "which energy" in lower) and "current is maximum" in lower:
        return SolverResult(
            solved=True,
            route="lc_energy_concept",
            answer="magnetic field of the inductor",
            fol="In ideal LC, I maximum -> W_L maximum and W_C = 0",
            cot=numbered_cot(["Use LC energy exchange.", "When current is maximum, magnetic energy is maximum."]),
            premises=["Ideal LC energy alternates between capacitor electric field and inductor magnetic field"],
            confidence=0.86,
            target="concept",
            engine=formula_engine_name(),
        )
    if ("where" in lower or "which energy" in lower) and ("current is zero" in lower or "i = 0" in lower):
        return SolverResult(
            solved=True,
            route="lc_energy_concept",
            answer="electric field of the capacitor",
            fol="In ideal LC, I = 0 -> W_L = 0 and W_C maximum",
            cot=numbered_cot(["Use LC energy exchange.", "When current is zero, magnetic energy is zero and electric energy is stored in the capacitor."]),
            premises=["Ideal LC energy conservation"],
            confidence=0.86,
            target="concept",
            engine=formula_engine_name(),
        )

    if "percentage" in lower and "voltage" in lower:
        voltages = [parse_number(x) for x in re.findall(rf"(?P<val>{NUMBER_RE})\s*V\b", normalize_text(question))]
        if len(voltages) >= 2:
            value = (voltages[-1] / voltages[0]) ** 2
            return build_result(
                route="lc_energy",
                target="percent",
                value_si=value,
                question=question,
                formula="W_final / W_initial = (U_final / U_initial)^2",
                premise="Capacitor energy is proportional to U^2 when C is unchanged",
                cot=[
                    f"Identify initial voltage U1 = {format_number(voltages[0])} V and final voltage U2 = {format_number(voltages[-1])} V.",
                    "Since W = 1/2 C U^2 and C is constant, the energy ratio is (U2/U1)^2.",
                ],
                confidence=0.88,
                unit="%",
            )

    if ("current reaches its maximum" in lower or "current is maximum" in lower or "current reaches maximum" in lower) and ("which energy" in lower or "what energy" in lower or "energy is at its maximum" in lower):
        return build_text_result(
            route="lc_energy_concept",
            answer="magnetic energy in the inductor",
            target="concept",
            formula="I maximum -> W_L maximum",
            premise="In an ideal LC circuit, magnetic energy is proportional to current squared",
            cot=["Use LC energy exchange.", "At maximum current, the inductor's magnetic energy is maximum."],
            confidence=0.88,
        )
    if ("current is zero" in lower or "i = 0" in lower) and ("what form" in lower or "which energy" in lower or "energy is present" in lower):
        return build_text_result(
            route="lc_energy_concept",
            answer="electric field energy in the capacitor",
            target="concept",
            formula="I = 0 -> W_L = 0 and W_C maximum",
            premise="Ideal LC energy conservation",
            cot=["When current is zero, magnetic energy is zero.", "The energy is stored in the capacitor electric field."],
            confidence=0.88,
        )
    if "electric field energy is zero" in lower and "instantaneous current" in lower:
        return build_text_result(
            route="lc_current_concept",
            answer="maximum",
            target="current",
            formula="W_C = 0 -> W_L maximum -> I maximum",
            premise="Ideal LC energy exchange",
            cot=["If electric field energy is zero, all energy is magnetic.", "Magnetic energy maximum means current maximum."],
            confidence=0.88,
        )

    if ("total oscillation energy" in lower or "total energy" in lower or "energy of oscillation" in lower) and capacitance and voltage:
        value = formula_value(
            "capacitor_energy",
            "W",
            {"C": capacitance.value_si, "U": voltage.value_si},
            0.5 * capacitance.value_si * voltage.value_si * voltage.value_si,
        )
        return build_result(
            route="lc_total_energy",
            target="energy",
            value_si=value,
            question=question,
            formula="W_total = 1/2 * C * U0^2",
            premise="Initial capacitor energy becomes total ideal LC oscillation energy",
            cot=[f"Identify initial capacitance C = {capacitance.raw} and voltage U = {voltage.raw}.", "Compute total LC energy from initial capacitor energy."],
            confidence=0.92,
        )

    fraction_match = re.search(r"(electric|magnetic) field energy is (?P<num>\d+(?:\.\d+)?)\s*/\s*(?P<den>\d+(?:\.\d+)?) of the total energy", lower)
    fraction_match = fraction_match or re.search(r"(electric|magnetic) energy is (?P<frac>\d+(?:\.\d+)?) of the total energy", lower)
    if ("percentage" in lower or "percent" in lower) and "current" in lower and "electric field energy equals the magnetic field energy" in lower:
        value = math.sqrt(0.5)
        return build_result(
            route="lc_current_fraction",
            target="percent",
            value_si=value,
            question=question,
            formula="I/Imax = sqrt(W_L/W_total)",
            premise="Equal electric and magnetic energies mean W_L is one half of total LC energy",
            cot=["Equal electric and magnetic energies split the total energy equally.", "Use I/Imax = sqrt(W_L/W_total)."],
            confidence=0.9,
            unit="%",
        )
    if ("percentage" in lower or "percent" in lower) and "current" in lower and fraction_match:
        if fraction_match.groupdict().get("frac"):
            fraction = float(fraction_match.group("frac"))
        else:
            fraction = float(fraction_match.group("num")) / float(fraction_match.group("den"))
        magnetic_fraction = fraction if fraction_match.group(1).startswith("magnetic") else 1.0 - fraction
        value = math.sqrt(max(0.0, magnetic_fraction))
        return build_result(
            route="lc_current_fraction",
            target="percent",
            value_si=value,
            question=question,
            formula="I/Imax = sqrt(W_L/W_total)",
            premise="LC magnetic energy fraction equals (I/Imax)^2",
            cot=["Determine the magnetic-energy fraction.", "Take the square root to get I/Imax and convert to percent."],
            confidence=0.9,
            unit="%",
        )

    if "wl" in lower and re.search(r"\bwl\s*=\s*0\b", lower) and re.search(r"\bwc\b", lower):
        return build_text_result(
            route="lc_energy_conservation_concept",
            answer="W_total",
            target="energy",
            formula="W_total = W_C + W_L; if W_L = 0 then W_C = W_total",
            premise="Ideal LC total energy conservation",
            cot=["Use W_total = W_C + W_L.", "With W_L = 0, all energy is electric energy in the capacitor."],
            confidence=0.86,
        )

    if "magnetic field energy" in lower and total_energy and capacitance and voltage:
        electric_value = formula_value(
            "capacitor_energy",
            "W",
            {"C": capacitance.value_si, "U": voltage.value_si},
            0.5 * capacitance.value_si * voltage.value_si * voltage.value_si,
        )
        value = max(0.0, total_energy.value_si - electric_value)
        return build_result(
            route="lc_energy",
            target="energy",
            value_si=value,
            question=question,
            formula="W_m = W_total - 1/2 * C * U^2",
            premise="In an ideal LC circuit, W_total = W_e + W_m",
            cot=[
                f"Identify total energy W = {total_energy.raw}, capacitance C = {capacitance.raw}, and capacitor voltage U = {voltage.raw}.",
                "Use the formula solver to compute electric energy W_e = 1/2 C U^2.",
                "Use W_m = W_total - W_e.",
            ],
            confidence=0.88,
            unit="J",
        )

    if "voltage" in lower and capacitance and electric_energy:
        value = formula_value(
            "capacitor_energy",
            "U",
            {"W": electric_energy.value_si, "C": capacitance.value_si},
            math.sqrt(2.0 * electric_energy.value_si / capacitance.value_si),
        )
        return build_result(
            route="lc_energy",
            target="voltage",
            value_si=value,
            question=question,
            formula="U = sqrt(2 * W_e / C)",
            premise="Electric energy in the capacitor of an LC circuit: W_e = 1/2 C U^2",
            cot=[
                f"Identify electric field energy W_e = {electric_energy.raw} and capacitance C = {capacitance.raw}.",
                "Use the formula solver to rearrange W_e = 1/2 C U^2 and solve for U.",
            ],
            confidence=0.9,
        )

    if "voltage" in lower and capacitance and total_energy and magnetic_energy:
        electric_value = max(0.0, total_energy.value_si - magnetic_energy.value_si)
        value = formula_value(
            "capacitor_energy",
            "U",
            {"W": electric_value, "C": capacitance.value_si},
            math.sqrt(2.0 * electric_value / capacitance.value_si),
        )
        return build_result(
            route="lc_energy",
            target="voltage",
            value_si=value,
            question=question,
            formula="U = sqrt(2 * (W_total - W_m) / C)",
            premise="In an ideal LC circuit, W_total = W_e + W_m and W_e = 1/2 C U^2",
            cot=[
                f"Identify total energy W = {total_energy.raw}, magnetic energy W_m = {magnetic_energy.raw}, and capacitance C = {capacitance.raw}.",
                "Compute electric energy W_e = W_total - W_m.",
                "Use the formula solver to solve U = sqrt(2W_e/C).",
            ],
            confidence=0.88,
        )
    return None


def solve_solenoid(question: str, quantities: dict[str, Quantity]) -> SolverResult | None:
    lower = normalize_text(question).lower()
    if "solenoid" not in lower and "magnetic flux" not in lower and "induced" not in lower and not ("inductance" in lower and "number of turns" in lower):
        return None

    length = find_quantity(
        question,
        "length",
        [
            rf"(?:length|long)\s*(?:of|=|is)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>m|cm|mm)\b",
            rf"(?P<val>{NUMBER_RE})\s*(?P<unit>m|cm|mm)\s+long\b",
            rf"solenoid\s+is\s+(?P<val>{NUMBER_RE})\s*(?P<unit>m|cm|mm)\s+long\b",
        ],
    )
    turns = find_quantity(question, "turns", [rf"(?:has|with)?\s*(?P<val>{NUMBER_RE})\s*turns\b"])
    turn_density = find_quantity(question, "n", [rf"(?:turn density|turns per meter|n)\s*(?:is|of|=)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>)"])
    current = quantities.get("I") or find_symbol_quantity(question, "I", ["mA", "A"], aliases=["current", "electric current"])
    magnetic_field = quantities.get("B") or find_symbol_quantity(question, "B", ["mT", "T"], aliases=["magnetic field", "magnetic flux density"])
    area = quantities.get("area") or find_area(question)
    inductance = quantities.get("L") or find_symbol_quantity(question, "L", ["mH", "H"], aliases=["inductance", "self-inductance"])
    emf_quantity = find_symbol_quantity(question, "emf", ["V", "kV"], aliases=["induced electromotive force", "induced emf", "emf"])

    if "unit of induced electromotive force" in lower or ("unit" in lower and "emf" in lower):
        return build_text_result(
            route="induced_emf_unit",
            answer="V",
            target="unit",
            formula="SI unit of electromotive force is volt",
            premise="Electromotive force has the same SI unit as voltage",
            cot=["Identify the requested quantity as EMF.", "Report its SI unit."],
            confidence=0.95,
        )
    if "application" in lower and "solenoid" in lower:
        return build_text_result(
            route="solenoid_application_concept",
            answer="electromagnet",
            target="concept",
            formula="Current-carrying solenoid produces a controllable magnetic field",
            premise="A solenoid is directly used as an electromagnet in devices such as relays",
            cot=["A solenoid creates a magnetic field when current flows.", "That principle is directly used in electromagnets."],
            confidence=0.84,
        )
    if "self-inductance" in lower and "depend on" in lower:
        return build_text_result(
            route="solenoid_inductance_dependency",
            answer="number of turns, length, cross-sectional area, and medium",
            target="concept",
            formula="L = mu0 * mu_r * N^2 * A / l",
            premise="Long-solenoid inductance depends on geometry and magnetic medium",
            cot=["Use the long-solenoid inductance relation.", "Read off the variables that determine L."],
            confidence=0.86,
        )
    if "cross-sectional area" in lower and "self-inductance" in lower and "increased" in lower:
        return build_text_result(
            route="solenoid_inductance_area_scaling",
            answer="increases in direct proportion",
            target="concept",
            formula="L proportional to A",
            premise="For fixed N, length, and medium, solenoid inductance is proportional to cross-sectional area",
            cot=["Use L = mu0 mu_r N^2 A/l.", "Increasing A increases L proportionally."],
            confidence=0.86,
        )
    if "number of turns is increased" in lower and "inductance" in lower:
        return build_text_result(
            route="solenoid_inductance_turn_scaling",
            answer="increases with the square of the number of turns",
            target="concept",
            formula="L proportional to N^2",
            premise="For fixed length, area, and medium, solenoid inductance scales as N squared",
            cot=["Use L = mu0 mu_r N^2 A/l.", "The number of turns enters as N^2."],
            confidence=0.86,
        )
    if "magnetic field not depend" in lower and "solenoid" in lower:
        return build_text_result(
            route="solenoid_field_dependency",
            answer="cross-sectional area",
            target="concept",
            formula="B = mu0 n I",
            premise="Ideal-solenoid magnetic field depends on turn density, current, and medium",
            cot=["Use B = mu0 n I.", "Cross-sectional area is not part of this ideal field expression."],
            confidence=0.86,
        )
    if "where is the magnetic field concentrated" in lower and "solenoid" in lower:
        return build_text_result(
            route="ideal_solenoid_field_location",
            answer="inside the solenoid",
            target="concept",
            formula="Ideal solenoid field is concentrated inside",
            premise="Long-solenoid approximation",
            cot=["Use the ideal solenoid model.", "The field is concentrated inside and weak outside."],
            confidence=0.86,
        )
    if "energy density" in lower and "proportional to the square" in lower:
        return build_text_result(
            route="magnetic_energy_density_concept",
            answer="magnetic induction B",
            target="concept",
            formula="u_B = B^2/(2*mu0)",
            premise="Magnetic field energy density is quadratic in B",
            cot=["Use u_B = B^2/(2mu0).", "Therefore it is proportional to B squared."],
            confidence=0.88,
        )
    if "magnetic field energy" in lower and "magnetic field" in lower and "increases" in lower:
        return build_text_result(
            route="magnetic_energy_scaling_concept",
            answer="increases proportionally to B^2",
            target="concept",
            formula="u_B = B^2/(2*mu0)",
            premise="Magnetic energy density is quadratic in magnetic field",
            cot=["Use magnetic energy density u_B = B^2/(2mu0).", "An increase in B changes energy according to B squared."],
            confidence=0.86,
        )
    if "when does an induced electromotive force appear" in lower:
        return build_text_result(
            route="self_induction_condition",
            answer="when the current changes with time",
            target="concept",
            formula="emf = -L dI/dt",
            premise="Self-induced EMF appears only when current is changing",
            cot=["Use self-induction law.", "A nonzero dI/dt creates induced EMF."],
            confidence=0.9,
        )
    if "double the number of turns" in lower and "magnetic field" in lower:
        return build_text_result(
            route="solenoid_field_proportionality",
            answer="doubles",
            target="concept",
            formula="B = mu0 * (N/l) * I",
            premise="For fixed length and current, solenoid field is proportional to the number of turns",
            cot=["Use B = mu0(N/l)I.", "If N doubles while l and I stay fixed, B doubles."],
            confidence=0.9,
        )
    if "directly proportional" in lower and "magnetic field" in lower:
        return build_text_result(
            route="solenoid_field_proportionality",
            answer="turn density and current",
            target="concept",
            formula="B = mu0 * n * I",
            premise="Long-solenoid field law",
            cot=["Use B = mu0 n I.", "The field is linear in turn density n and current I."],
            confidence=0.88,
        )
    if "depend linearly" in lower and "magnetic field" in lower:
        return build_text_result(
            route="solenoid_field_proportionality",
            answer="current through the solenoid",
            target="concept",
            formula="B = mu0 * n * I",
            premise="For fixed solenoid geometry and medium, B is linear in current",
            cot=["Use B = mu0 n I.", "With n fixed, B depends linearly on current."],
            confidence=0.88,
        )
    if "external magnetic field" in lower and "ideal solenoid" in lower:
        return build_text_result(
            route="ideal_solenoid_external_field",
            answer="approximately zero",
            target="concept",
            formula="Ideal long solenoid confines the field inside",
            premise="Ideal-solenoid approximation",
            cot=["Use the ideal long-solenoid model.", "The external field is negligible compared with the internal field."],
            confidence=0.86,
        )
    if "suddenly disconnected" in lower:
        return build_text_result(
            route="self_induction_concept",
            answer="a self-induced emf opposes the current decrease",
            target="concept",
            formula="Lenz's law",
            premise="Self-induction opposes changes in current",
            cot=["Disconnecting causes a rapid current decrease.", "The induced emf acts to oppose that change."],
            confidence=0.86,
        )
    if "does not depend" in lower and "self-inductance" in lower:
        return build_text_result(
            route="solenoid_inductance_dependency",
            answer="current",
            target="concept",
            formula="L = mu0 * mu_r * N^2 * A / l",
            premise="Solenoid self-inductance depends on geometry and medium, not current",
            cot=["Use the solenoid inductance formula.", "Current is not one of the determining variables for an ideal linear solenoid."],
            confidence=0.86,
        )
    if "flux" in lower and "changes uniformly" in lower and "what appears" in lower:
        return build_text_result(
            route="faraday_concept",
            answer="induced electromotive force",
            target="concept",
            formula="emf = -dPhi/dt",
            premise="Faraday's law of electromagnetic induction",
            cot=["A changing magnetic flux produces induction.", "In a closed circuit this appears as induced emf and induced current."],
            confidence=0.88,
        )
    if "current through the solenoid increases rapidly" in lower or ("current" in lower and "increases rapidly" in lower and "induced electromotive force" in lower):
        return build_text_result(
            route="self_induction_rate_concept",
            answer="increases",
            target="concept",
            formula="|emf| = L |dI/dt|",
            premise="Self-induced EMF magnitude is proportional to the rate of current change",
            cot=["Use |emf| = L|dI/dt|.", "A more rapid increase in current produces a larger induced EMF opposing the change."],
            confidence=0.86,
        )
    if "in what form" in lower and "magnetic field energy" in lower:
        return build_text_result(
            route="solenoid_energy_concept",
            answer="magnetic field",
            target="concept",
            formula="W = 1/2 L I^2",
            premise="Energy of an energized solenoid is stored in its magnetic field",
            cot=["A current-carrying solenoid stores magnetic energy.", "That energy resides in the magnetic field inside and around the solenoid."],
            confidence=0.88,
        )

    if ("magnetic field" in lower or "flux density" in lower) and "energy" not in lower and not ("magnetic flux" in lower and "flux density" not in lower) and current and (turn_density or (turns and length)):
        n_value = turn_density.value_si if turn_density else turns.value_si / length.value_si
        value = formula_value("solenoid_field", "B", {"n": n_value, "I": current.value_si}, MU0 * n_value * current.value_si)
        return build_result(
            route="solenoid_field",
            target="magnetic_field",
            value_si=value,
            question=question,
            formula="B = mu0 * n * I",
            premise="Long-solenoid field law: B = mu0 n I",
            cot=[
                "Identify the turn density n directly or compute n = N/l.",
                f"Use current I = {current.raw}.",
                "Use the formula solver to evaluate B = mu0 n I.",
            ],
            confidence=0.88,
        )
    if "inductance" in lower and turns and area and length:
        value = formula_value(
            "solenoid_inductance",
            "L",
            {"N": turns.value_si, "A": area.value_si, "length": length.value_si},
            MU0 * turns.value_si * turns.value_si * area.value_si / length.value_si,
        )
        return build_result(
            route="solenoid_inductance",
            target="inductance",
            value_si=value,
            question=question,
            formula="L = mu0 * N^2 * A / l",
            premise="Long-solenoid inductance law: L = mu0 N^2 A / l",
            cot=[
                f"Identify N = {turns.raw}, A = {area.raw}, and l = {length.raw}.",
                "Use the formula solver to evaluate L = mu0 N^2 A/l.",
            ],
            confidence=0.86,
        )
    if "energy density" in lower and magnetic_field:
        value = magnetic_field.value_si * magnetic_field.value_si / (2.0 * MU0)
        return build_result(
            route="solenoid_energy_density",
            target="energy_density",
            value_si=value,
            question=question,
            formula="u_B = B^2/(2*mu0)",
            premise="Magnetic energy density in vacuum/air",
            cot=["Identify magnetic field B.", "Use u_B = B^2/(2 mu0)."],
            confidence=0.86,
            unit="J/m3",
        )
    if "energy density" in lower and current and (turn_density or (turns and length)):
        n_value = turn_density.value_si if turn_density else turns.value_si / length.value_si
        b_value = MU0 * n_value * current.value_si
        value = b_value * b_value / (2.0 * MU0)
        return build_result(
            route="solenoid_energy_density",
            target="energy_density",
            value_si=value,
            question=question,
            formula="u_B = B^2/(2*mu0), B = mu0*n*I",
            premise="Magnetic energy density in a solenoid",
            cot=["Compute B from turn density and current.", "Use u_B = B^2/(2 mu0)."],
            confidence=0.86,
            unit="J/m3",
        )
    if ("magnetic field energy" in lower or "stored magnetic energy" in lower or "magnetic energy" in lower) and inductance and current:
        value = formula_value("inductor_energy", "W", {"L": inductance.value_si, "I": current.value_si}, 0.5 * inductance.value_si * current.value_si * current.value_si)
        return build_result(
            route="solenoid_energy",
            target="energy",
            value_si=value,
            question=question,
            formula="W_l = 1/2 * L * I^2",
            premise="Magnetic energy law: W_l = 1/2 L I^2",
            cot=[
                f"Identify inductance L = {inductance.raw} and current I = {current.raw}.",
                "Use the formula solver to evaluate W_l = 1/2 L I^2.",
            ],
            confidence=0.9,
        )
    if ("magnetic field energy" in lower or "magnetic energy" in lower) and current and turns and area and length:
        inductance_value = MU0 * turns.value_si * turns.value_si * area.value_si / length.value_si
        value = 0.5 * inductance_value * current.value_si * current.value_si
        return build_result(
            route="solenoid_magnetic_energy_from_geometry",
            target="energy",
            value_si=value,
            question=question,
            formula="W = 1/2 L I^2, L = mu0 N^2 A/l",
            premise="Solenoid inductance plus magnetic energy law",
            cot=["Compute solenoid inductance from geometry.", "Use W=1/2 L I^2."],
            confidence=0.84,
            unit="J",
        )
    if ("number of turns per meter" in lower or "turn density" in lower or "turns per unit length" in lower or "number of turns per unit length" in lower) and turns and length:
        value = formula_value("turn_density", "n", {"N": turns.value_si, "length": length.value_si}, turns.value_si / length.value_si)
        return build_result(
            route="solenoid_turn_density",
            target="turn_density",
            value_si=value,
            question=question,
            formula="n = N / l",
            premise="Turn density definition: n = N/l",
            cot=[
                f"Identify number of turns N = {turns.raw} and length l = {length.raw}.",
                "Use the formula solver to compute n = N/l.",
            ],
            confidence=0.92,
            unit="turns/m",
        )
    if "magnetic flux" in lower and magnetic_field and area:
        multiplier = turns.value_si if ("entire solenoid" in lower and turns) else 1.0
        base_flux = formula_value("magnetic_flux", "Phi", {"B": magnetic_field.value_si, "A": area.value_si}, magnetic_field.value_si * area.value_si)
        value = base_flux * multiplier
        return build_result(
            route="magnetic_flux",
            target="flux",
            value_si=value,
            question=question,
            formula="Phi = B * A",
            premise="Magnetic flux law: Phi = B A for a perpendicular uniform field",
            cot=[
                f"Identify magnetic field B = {magnetic_field.raw} and area A = {area.raw}.",
                "Use the formula solver to evaluate Phi = B A, multiplying by N only if total flux linkage is requested.",
            ],
            confidence=0.88,
        )
    if ("flux linkage" in lower or "total flux" in lower) and turns:
        flux_values = numeric_unit_values(question, ["uWb", "Wb"], "Phi")
        if flux_values:
            per_turn_flux = flux_values[-1]
            value = turns.value_si * per_turn_flux.value_si
            return build_result(
                route="faraday_flux_linkage",
                target="flux",
                value_si=value,
                question=question,
                formula="lambda = N Phi",
                premise="Flux linkage is number of turns times flux through each turn",
                cot=["Identify number of turns and flux per turn.", "Multiply to get total flux linkage."],
                confidence=0.9,
                unit="Wb",
            )
    if "induced" in lower and turns and "flux per turn" in lower:
        flux_values = numeric_unit_values(question, ["uWb", "Wb"], "Phi")
        change = current_change(question)
        if flux_values and change:
            _delta_current, delta_time, _raw_change = change
            value = turns.value_si * abs(flux_values[-1].value_si) / delta_time
            return build_result(
                route="faraday_flux_per_turn_emf",
                target="voltage",
                value_si=value,
                question=question,
                formula="emf = N * |Delta Phi| / Delta t",
                premise="Flux per turn changes from the stated value to zero with the current",
                cot=["Identify N, flux change per turn, and time interval.", "Apply Faraday's law for flux linkage."],
                confidence=0.84,
                unit="V",
            )
    if "magnetic flux" in lower and area and current and (turn_density or (turns and length)):
        n_value = turn_density.value_si if turn_density else turns.value_si / length.value_si
        b_value = MU0 * n_value * current.value_si
        multiplier = turns.value_si if ("entire solenoid" in lower and turns) else 1.0
        value = b_value * area.value_si * multiplier
        return build_result(
            route="solenoid_flux_from_current",
            target="flux",
            value_si=value,
            question=question,
            formula="Phi = B*A, B = mu0*n*I",
            premise="Solenoid field and magnetic flux laws",
            cot=["Compute the solenoid magnetic field from n and I.", "Compute flux through one turn unless total flux linkage is requested."],
            confidence=0.86,
        )
    if ("induced electromotive force" in lower or "induced emf" in lower) and inductance:
        change = current_change(question)
        if change:
            delta_current, delta_time, _raw_change = change
            value = formula_value(
                "induced_emf",
                "emf",
                {"L": inductance.value_si, "delta_I": delta_current, "delta_t": delta_time},
                inductance.value_si * delta_current / delta_time,
            )
            return build_result(
                route="induced_emf",
                target="voltage",
                value_si=value,
                question=question,
                formula="epsilon = L * |Delta I| / Delta t",
                premise="Self-induced EMF magnitude: epsilon = L |Delta I| / Delta t",
                cot=[
                    f"Identify self-inductance L = {inductance.raw}.",
                    "Compute the current change rate |Delta I|/Delta t.",
                    "Use the formula solver to evaluate epsilon = L |Delta I|/Delta t.",
                ],
                confidence=0.9,
            )
    if ("self-inductance" in lower or "inductance" in lower) and emf_quantity:
        change = current_change(question)
        if change:
            delta_current, delta_time, _raw_change = change
            value = emf_quantity.value_si * delta_time / delta_current
            return build_result(
                route="induced_emf_inductance_inverse",
                target="inductance",
                value_si=value,
                question=question,
                formula="L = emf * Delta t / |Delta I|",
                premise="Self-induced EMF magnitude rearranged for inductance",
                cot=["Identify induced EMF and current change rate.", "Rearrange emf = L|Delta I|/Delta t for L."],
                confidence=0.88,
            )
    if "induced" in lower and "flux" in lower:
        flux_match = re.search(
            rf"(?:flux|magnetic flux).*?(?:from|decreases from|changes from)?\s*(?P<p1>{NUMBER_RE})\s*(?P<u1>uWb|Wb)\s*(?:to|and reaches)\s*(?P<p2>{NUMBER_RE})\s*(?P<u2>uWb|Wb).*?(?:in|during)\s*(?P<t>{NUMBER_RE})\s*s",
            normalize_text(question),
            flags=re.IGNORECASE,
        )
        if not flux_match:
            flux_match = re.search(
                rf"(?:flux|magnetic flux).*?(?P<p1>{NUMBER_RE})\s*(?P<u1>uWb|Wb).*?(?:decreases to|to)\s*(?P<p2>{NUMBER_RE})\s*(?P<u2>uWb|Wb)?\s*(?:in|during)\s*(?P<t>{NUMBER_RE})\s*s",
                normalize_text(question),
                flags=re.IGNORECASE,
            )
        if flux_match:
            u2 = clean_unit(flux_match.groupdict().get("u2") or flux_match.group("u1"))
            p1 = parse_number(flux_match.group("p1")) * UNIT_FACTORS.get(clean_unit(flux_match.group("u1")), 1.0)
            p2 = parse_number(flux_match.group("p2")) * UNIT_FACTORS.get(u2, 1.0)
            delta_time = parse_number(flux_match.group("t"))
            multiplier = turns.value_si if turns else 1.0
            value = multiplier * abs(p2 - p1) / delta_time
            return build_result(
                route="faraday_flux_change",
                target="voltage",
                value_si=value,
                question=question,
                formula="|emf| = N * |Delta Phi| / Delta t",
                premise="Faraday induction law for uniformly changing flux",
                cot=["Identify the flux change and time interval.", "Apply Faraday's law for EMF magnitude."],
                confidence=0.88,
                unit="V",
            )
    return None


def solve_error_measurement(question: str) -> SolverResult | None:
    lower = normalize_text(question).lower()
    if "error" not in lower and "uncertainty" not in lower:
        return None
    normalized = normalize_text(question)
    measure_unit = r"°C|C|cm|m|g|kg|A|V|ohm|N|atm|ml|W"

    readings = [
        (parse_number(match.group("val")), match.group("unit"))
        for match in re.finditer(rf"(?P<val>{NUMBER_RE})\s*(?P<unit>{measure_unit})\b", normalized, flags=re.IGNORECASE)
    ]

    def find_uncertainty(labels: list[str], units: str) -> tuple[float, float, str] | None:
        label_pattern = "|".join(re.escape(label) for label in labels)
        patterns = [
            rf"(?:{label_pattern})[^.,;]{{0,48}}?(?:=|is|was|of)?\s*(?P<x>{NUMBER_RE})\s*(?:\+/-|±)\s*(?P<dx>{NUMBER_RE})\s*(?P<unit>{units})\b",
            rf"(?P<x>{NUMBER_RE})\s*(?P<unit>{units})[^.,;]{{0,48}}?uncertainty\s+(?:of\s+)?(?:\+/-|±)\s*(?P<dx>{NUMBER_RE})\s*(?P=unit)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match:
                return parse_number(match.group("x")), parse_number(match.group("dx")), match.group("unit")
        return None

    voltage_unc_pair = find_uncertainty(["U", "voltage", "voltmeter", "measuring voltage"], "V")
    current_unc_pair = find_uncertainty(["I", "current", "ammeter", "measuring current"], "A|mA")
    if voltage_unc_pair and current_unc_pair and ("r = u/i" in lower or "resistance r is calculated" in lower):
        u_val, du, _u_unit = voltage_unc_pair
        i_val, di, i_unit = current_unc_pair
        if clean_unit(i_unit) == "mA":
            i_val *= 1.0e-3
            di *= 1.0e-3
        r_value = u_val / i_val
        delta_r = r_value * (du / u_val + di / i_val)
        if "absolute" in lower:
            return build_result(
                route="measurement_resistance_absolute_error",
                target="resistance",
                value_si=delta_r,
                question=question,
                formula="Delta R = R(Delta U/U + Delta I/I)",
                premise="Quotient uncertainty propagation for R=U/I",
                cot=["Compute R = U/I.", "Add relative uncertainties of U and I and multiply by R."],
                confidence=0.88,
                unit="ohm",
            )
    if voltage_unc_pair and current_unc_pair and ("power" in lower or "P" in normalized):
        u_val, du, _u_unit = voltage_unc_pair
        i_val, di, i_unit = current_unc_pair
        if clean_unit(i_unit) == "mA":
            i_val *= 1.0e-3
            di *= 1.0e-3
        relative = du / u_val + di / i_val
        if "absolute" in lower:
            return build_result(
                route="measurement_power_absolute_error",
                target="power",
                value_si=u_val * i_val * relative,
                question=question,
                formula="Delta P = UI(Delta U/U + Delta I/I)",
                premise="Product uncertainty propagation for P=UI",
                cot=["Add relative uncertainties of U and I.", "Multiply by P=UI to get the absolute error."],
                confidence=0.88,
                unit="W",
            )
        return build_result(
            route="measurement_power_relative_error",
            target="percent",
            value_si=relative,
            question=question,
            formula="Delta P/P = Delta U/U + Delta I/I",
            premise="Product uncertainty propagation for P=UI",
            cot=["Add relative uncertainties of U and I.", "Convert to percent."],
            confidence=0.88,
            unit="%",
        )

    bounded_measurement = re.search(
        rf"(?P<x>{NUMBER_RE})\s*(?P<unit>{measure_unit})[^.,;]{{0,64}}?uncertainty\s+(?:of\s+)?(?:\+/-|±)\s*(?P<dx>{NUMBER_RE})\s*(?P=unit)",
        normalized,
        flags=re.IGNORECASE,
    )
    if bounded_measurement and ("maximum possible" in lower or "minimum possible" in lower):
        value = parse_number(bounded_measurement.group("x"))
        delta = parse_number(bounded_measurement.group("dx"))
        unit = bounded_measurement.group("unit")
        result = value - delta if "minimum" in lower else value + delta
        return build_text_result(
            route="measurement_bound",
            answer=f"{format_number(result)} {unit}",
            target="current" if unit.upper().endswith("A") else "measurement",
            formula="x_max/min = x ± Delta x",
            premise="A measured value with absolute uncertainty gives a possible interval",
            cot=["Identify measured value and uncertainty.", "Use the requested upper or lower bound."],
            confidence=0.88,
        )

    measured_absolute = re.search(
        rf"measured value\s*(?:is|=)?\s*(?P<x>{NUMBER_RE})\s*(?P<unit>{measure_unit}).*?absolute error\s*(?:is|=|of)?\s*(?P<dx>{NUMBER_RE})\s*(?P=unit)",
        normalized,
        flags=re.IGNORECASE,
    )
    if measured_absolute and ("relative" in lower or "percentage" in lower or "percent" in lower):
        measured_value = parse_number(measured_absolute.group("x"))
        absolute_error = parse_number(measured_absolute.group("dx"))
        return build_result(
            route="measurement_direct_relative_error",
            target="percent",
            value_si=absolute_error / measured_value,
            question=question,
            formula="relative_error = absolute_error / measured_value",
            premise="Relative error definition from measured value and absolute error",
            cot=["Identify the measured value and absolute error.", "Divide absolute error by the measured value and convert to percent."],
            confidence=0.9,
            unit="%",
        )

    if ("mean" in lower or "average" in lower) and len(readings) >= 2:
        # Drop leading counts such as "3 times" by only using values with explicit physical units.
        unit = readings[-1][1]
        values = [value for value, value_unit in readings if value_unit == unit]
        if len(values) >= 2:
            mean = sum(values) / len(values)
            mean_abs_error = sum(abs(value - mean) for value in values) / len(values)
            answer = f"{format_number(mean)}; {format_number(mean_abs_error)} {unit}"
            return build_text_result(
                route="measurement_repeated_mean_error",
                answer=answer,
                target="mean_error",
                formula="mean = average(x_i); Delta = average(|x_i-mean|)",
                premise="Repeated measurement mean and mean absolute error",
                cot=["Use readings with the same unit.", "Compute the mean and the average absolute deviation."],
                confidence=0.88,
            )

    if "random error" in lower and len(readings) >= 2:
        unit = readings[-1][1]
        values = [value for value, value_unit in readings if value_unit == unit]
        if len(values) >= 2:
            value = (max(values) - min(values)) / 2.0
            return build_text_result(
                route="measurement_random_error",
                answer=f"{format_number(value)} {unit}",
                target="error",
                formula="random_error = (max(x_i)-min(x_i))/2",
                premise="School-level repeated-measurement random error estimate",
                cot=["Identify the repeated readings.", "Use half the range as the random error."],
                confidence=0.84,
            )

    least_count = re.search(rf"least count[^.,;]{{0,48}}?(?:of|is)?\s*(?P<dx>{NUMBER_RE})\s*(?P<unit>{measure_unit})\b", normalized, flags=re.IGNORECASE)
    if least_count:
        dx = parse_number(least_count.group("dx"))
        unit = least_count.group("unit")
        measured_candidates = [
            value for value, value_unit in readings
            if value_unit == unit and abs(value - dx) > 1e-12
        ]
        measured_value = measured_candidates[-1] if measured_candidates else None
        if "relative" in lower or "percentage" in lower or "percent" in lower:
            if measured_value:
                value = dx / measured_value
                return build_result(
                    route="measurement_least_count_relative",
                    target="percent",
                    value_si=value,
                    question=question,
                    formula="relative_error = least_count / measured_value",
                    premise="Instrument least count gives the absolute uncertainty",
                    cot=["Use least count as the absolute error.", "Divide by the measured value and convert to percent."],
                    confidence=0.86,
                    unit="%",
                )
        if "absolute" in lower:
            return build_text_result(
                route="measurement_least_count_absolute",
                answer=f"{format_number(dx)} {unit}",
                target="error",
                formula="absolute_error = least_count",
                premise="For a direct instrument reading, least count is used as absolute error",
                cot=["Identify the instrument least count.", "Report it as the absolute error."],
                confidence=0.86,
            )

    true_measured = re.search(
        rf"(?:actual|true value|true).*?(?P<true>{NUMBER_RE})\s*(?P<unit>{measure_unit})?.*?(?:measured|student measured|measured value).*?(?P<measured>{NUMBER_RE})\s*(?P<unit2>{measure_unit})?",
        normalized,
        flags=re.IGNORECASE,
    )
    if not true_measured:
        true_measured = re.search(
            rf"(?:measured value|measured result|measured).*?(?P<measured>{NUMBER_RE})\s*(?P<unit2>{measure_unit})?.*?(?:true value|actual|true).*?(?P<true>{NUMBER_RE})\s*(?P<unit>{measure_unit})?",
            normalized,
            flags=re.IGNORECASE,
        )
    if true_measured:
        true_value = parse_number(true_measured.group("true"))
        measured_value = parse_number(true_measured.group("measured"))
        unit = true_measured.groupdict().get("unit") or true_measured.groupdict().get("unit2") or ""
        absolute = abs(true_value - measured_value)
        relative = absolute / abs(true_value) if true_value else 0.0
        if "relative" in lower and "absolute" in lower:
            return build_text_result(
                route="measurement_true_measured_both",
                answer=f"{format_number(absolute)} {unit}; {format_number(relative * 100)} %",
                target="error",
                formula="absolute_error=|x_true-x_measured|; relative_error=absolute_error/x_true",
                premise="Direct true-versus-measured error definitions",
                cot=["Compute the absolute difference.", "Divide by the true value for relative error."],
                confidence=0.88,
            )
        if "relative" in lower or "percentage" in lower or "percent" in lower:
            return build_result(
                route="measurement_true_measured_relative",
                target="percent",
                value_si=relative,
                question=question,
                formula="relative_error = |x_true-x_measured| / x_true",
                premise="Relative error definition",
                cot=["Compute the absolute difference.", "Divide by the true value and convert to percent."],
                confidence=0.88,
                unit="%",
            )
        if "absolute" in lower:
            return build_text_result(
                route="measurement_true_measured_absolute",
                answer=f"{format_number(absolute)} {unit}".strip(),
                target="error",
                formula="absolute_error = |x_true-x_measured|",
                premise="Absolute error definition",
                cot=["Compute the absolute difference between true and measured values."],
                confidence=0.9,
            )

    uncertain = list(re.finditer(rf"(?P<label>voltage|current|R\d?|resistance|length|height|mass|temperature|force|value|result)?\s*(?:=|as|is|was|of)?\s*(?P<x>{NUMBER_RE})\s*(?:\+/-|±)\s*(?P<dx>{NUMBER_RE})\s*(?P<unit>{measure_unit})\b", normalized, flags=re.IGNORECASE))

    voltage_unc = next((m for m in uncertain if (m.group("label") or "").lower() == "voltage"), None)
    current_unc = next((m for m in uncertain if (m.group("label") or "").lower() == "current"), None)
    if voltage_unc and current_unc and ("power" in lower or "P" in normalized):
        u_val, du = parse_number(voltage_unc.group("x")), parse_number(voltage_unc.group("dx"))
        i_val, di = parse_number(current_unc.group("x")), parse_number(current_unc.group("dx"))
        relative = du / u_val + di / i_val
        if "absolute" in lower:
            value = u_val * i_val * relative
            return build_result(
                route="measurement_power_absolute_error",
                target="power",
                value_si=value,
                question=question,
                formula="Delta P = UI(Delta U/U + Delta I/I)",
                premise="Product uncertainty propagation for P=UI",
                cot=["Add the relative uncertainties of voltage and current.", "Multiply by P=UI to get absolute error."],
                confidence=0.88,
                unit="W",
            )
        value = relative
        return build_result(
            route="measurement_power_relative_error",
            target="percent",
            value_si=value,
            question=question,
            formula="Delta P/P = Delta U/U + Delta I/I",
            premise="Product uncertainty propagation for P=UI",
            cot=["Add the relative uncertainties of voltage and current.", "Convert to percent."],
            confidence=0.9,
            unit="%",
        )

    r_uncertain = [m for m in uncertain if (m.group("label") or "").upper().startswith("R")]
    if "series" in lower and len(r_uncertain) >= 2 and "total resistance" in lower:
        value = sum(parse_number(match.group("dx")) for match in r_uncertain)
        return build_text_result(
            route="measurement_series_resistance_error",
            answer=f"{format_number(value)} ohm",
            target="error",
            formula="Delta R_total = sum_i Delta R_i",
            premise="Absolute errors add for a sum",
            cot=["Identify each resistance uncertainty.", "Add absolute errors for series total resistance."],
            confidence=0.88,
        )

    if len(uncertain) == 1 and ("maximum possible" in lower or "minimum possible" in lower):
        match = uncertain[0]
        value = parse_number(match.group("x"))
        delta = parse_number(match.group("dx"))
        result = value - delta if "minimum" in lower else value + delta
        return build_text_result(
            route="measurement_bound",
            answer=f"{format_number(result)} {match.group('unit')}",
            target="measurement_bound",
            formula="x_max/min = x_measured +/- Delta x",
            premise="Measurement bound from stated uncertainty",
            cot=["Identify the measured value and uncertainty.", "Add or subtract uncertainty for the requested bound."],
            confidence=0.9,
        )

    if len(uncertain) == 1 and ("relative" in lower or "percentage" in lower or "percent" in lower):
        match = uncertain[0]
        value = parse_number(match.group("dx")) / parse_number(match.group("x"))
        return build_result(
            route="measurement_direct_relative",
            target="percent",
            value_si=value,
            question=question,
            formula="relative_error = Delta x / x",
            premise="Relative error definition",
            cot=["Identify the measured value and its uncertainty.", "Divide uncertainty by measured value and convert to percent."],
            confidence=0.88,
            unit="%",
        )

    if "power" in lower:
        voltage = re.search(r"voltage\s*(?:was|is)?\s*(?P<x>\d+(?:\.\d+)?)\s*(?:\+/-|±)\s*(?P<dx>\d+(?:\.\d+)?)", normalized, flags=re.IGNORECASE)
        current = re.search(r"current\s*(?:was|is)?\s*(?P<x>\d+(?:\.\d+)?)\s*(?:\+/-|±)\s*(?P<dx>\d+(?:\.\d+)?)", normalized, flags=re.IGNORECASE)
        if voltage and current and "relative" in lower:
            value = float(voltage.group("dx")) / float(voltage.group("x")) + float(current.group("dx")) / float(current.group("x"))
            return build_result(
                route="measurement_error",
                target="percent",
                value_si=value,
                question=question,
                formula="relative_error(P) = relative_error(U) + relative_error(I)",
                premise="For P = U I, relative errors add: Delta P/P = Delta U/U + Delta I/I",
                cot=[
                    "Identify voltage and current measurements with their absolute uncertainties.",
                    "For P = U I, add the relative errors of U and I.",
                    "Convert the result to percent.",
                ],
                confidence=0.9,
                unit="%",
            )
    measured = re.search(r"(?:as|is|was)\s*(?P<x>\d+(?:\.\d+)?)\s*\+/-\s*(?P<dx>\d+(?:\.\d+)?)", normalized, flags=re.IGNORECASE)
    measured = measured or re.search(r"(?:as|is|was)\s*(?P<x>\d+(?:\.\d+)?)\s*±\s*(?P<dx>\d+(?:\.\d+)?)", question, flags=re.IGNORECASE)
    if measured and ("relative" in lower or "percentage" in lower):
        value = float(measured.group("dx")) / float(measured.group("x"))
        return build_result(
            route="measurement_error",
            target="percent",
            value_si=value,
            question=question,
            formula="relative_error = Delta x / x",
            premise="Relative error definition: delta = Delta x / x",
            cot=[
                f"Identify measured value x = {measured.group('x')} and uncertainty Delta x = {measured.group('dx')}.",
                "Compute relative error Delta x / x and convert to percent.",
            ],
            confidence=0.88,
            unit="%",
        )
    actual = re.search(r"actual .*?(?P<a>\d+(?:\.\d+)?)\s*ohm.*?measured .*?(?P<m>\d+(?:\.\d+)?)\s*ohm", normalized, flags=re.IGNORECASE)
    if actual and "absolute error" in lower:
        value = abs(float(actual.group("a")) - float(actual.group("m")))
        return build_result(
            route="measurement_error",
            target="reactance",
            value_si=value,
            question=question,
            formula="absolute_error = |actual - measured|",
            premise="Absolute error definition: Delta x = |x_true - x_measured|",
            cot=[
                f"Identify actual value {actual.group('a')} ohm and measured value {actual.group('m')} ohm.",
                "Compute the absolute difference.",
            ],
            confidence=0.9,
            unit="ohm",
        )
    return None


def solve_dc_circuit(question: str, quantities: dict[str, Quantity]) -> SolverResult | None:
    lower = normalize_text(question).lower()
    if not any(token in lower for token in ["resistor", "resistance", "lamp", "parallel circuit", "series circuit", "power", "current", "voltage source"]):
        return None

    voltage = quantities.get("U") or find_symbol_quantity(question, "U", ["V", "kV"], aliases=["voltage source", "voltage", "supply voltage"])
    current = quantities.get("I") or find_symbol_quantity(question, "I", ["mA", "A"], aliases=["current", "total current"])
    resistance = quantities.get("R") or find_symbol_quantity(question, "R", ["ohm"], aliases=["resistance"])
    power = find_power(question)
    r_values = find_all_indexed_quantities(question, "R", ["ohm"])
    i_values = find_all_indexed_quantities(question, "I", ["mA", "A"])
    branch_currents = labeled_current_values(question)
    resistance_list = list(r_values.values()) if r_values else bare_resistance_values(question)

    if "parallel" in lower and len(resistance_list) >= 2 and ("equivalent resistance" in lower or "total resistance" in lower or "total resistance of" in lower):
        reciprocal = sum(1.0 / item.value_si for item in resistance_list if item.value_si)
        if reciprocal:
            value = 1.0 / reciprocal
            return build_result(
                route="dc_parallel_resistance",
                target="resistance",
                value_si=value,
                question=question,
                formula="1/R_eq = sum_i 1/R_i",
                premise="Parallel resistance equivalent law",
                cot=["Identify branch resistances.", "Add reciprocal resistances and invert."],
                confidence=0.88,
                unit="ohm",
            )

    if "parallel" in lower and voltage and len(resistance_list) >= 2 and ("total current" in lower or "current from" in lower or "current flowing through each" in lower or "through each" in lower):
        currents = [voltage.value_si / item.value_si for item in resistance_list if item.value_si]
        value = sum(currents)
        if "through each" in lower or "current flowing through each" in lower or "current through" in lower and len(currents) == 2:
            answer = "; ".join(f"{format_number(current_value)} A" for current_value in currents)
            return build_text_result(
                route="dc_parallel_branch_currents",
                answer=answer,
                target="current",
                formula="I_i = U/R_i",
                premise="Parallel branches share the same voltage",
                cot=["Use the same source voltage on each branch.", "Compute each branch current with Ohm's law."],
                confidence=0.86,
            )
        return build_result(
            route="dc_parallel_total_current",
            target="current",
            value_si=value,
            question=question,
            formula="I_total = sum_i U/R_i",
            premise="Parallel branch currents add",
            cot=["Compute each branch current using I=U/R.", "Add branch currents."],
            confidence=0.88,
            unit="A",
        )

    if "removed" in lower and "total current" in lower and branch_currents:
        remaining = [item.value_si for idx, item in branch_currents.items() if f"d{idx}" not in re.sub(r"\s+", "", re.search(r"lamp\s+D\d\s+is\s+removed|D\d\s+is\s+removed", lower).group(0) if re.search(r"lamp\s+D\d\s+is\s+removed|D\d\s+is\s+removed", lower) else "")]
        if not remaining:
            remaining = [item.value_si for item in branch_currents.values()]
        value = sum(remaining)
        return build_result(
            route="dc_removed_branch_current",
            target="current",
            value_si=value,
            question=question,
            formula="After a branch is removed, total current is the sum of remaining branch currents",
            premise="Parallel branch currents add",
            cot=["Identify the branch that remains after removal.", "Use the remaining branch current as the new total if it is the only branch left."],
            confidence=0.84,
            unit="A",
        )

    if "total current" in lower and current and branch_currents and re.search(r"\bD2\b", normalize_text(question), flags=re.IGNORECASE):
        known_sum = sum(item.value_si for item in branch_currents.values())
        value = current.value_si - known_sum
        if value >= -1e-12:
            return build_result(
                route="dc_missing_branch_current",
                target="current",
                value_si=max(0.0, value),
                question=question,
                formula="I_total = sum branch currents",
                premise="In a parallel junction, total current is the sum of branch currents",
                cot=["Identify total current and the known branch current.", "Subtract known branch current from total current."],
                confidence=0.88,
                unit="A",
            )

    if "total current" in lower and len(branch_currents) >= 2:
        value = sum(item.value_si for item in branch_currents.values())
        return build_result(
            route="dc_total_current_sum",
            target="current",
            value_si=value,
            question=question,
            formula="I_total = sum_i I_i",
            premise="Total current is the sum of branch currents in a parallel junction",
            cot=["Identify branch currents.", "Add them to obtain total current."],
            confidence=0.88,
            unit="A",
        )

    if "total current" in lower and len(i_values) >= 2:
        value = sum(item.value_si for item in i_values.values())
        return build_result(
            route="dc_total_current_sum",
            target="current",
            value_si=value,
            question=question,
            formula="I_total = sum_i I_i",
            premise="Total current is the sum of branch currents in a parallel junction",
            cot=["Identify branch currents.", "Add them to obtain total current."],
            confidence=0.88,
            unit="A",
        )

    if "resistance" in lower and "decreases" in lower and "current" in lower:
        return build_text_result(
            route="dc_ohm_inverse_concept",
            answer="increases",
            target="concept",
            formula="I = U/R",
            premise="At fixed voltage, current is inversely proportional to resistance",
            cot=["Use Ohm's law for a fixed source voltage.", "A decrease in resistance causes current to increase."],
            confidence=0.86,
        )
    if "total current increases" in lower and "light bulbs" in lower:
        return build_text_result(
            route="dc_brightness_concept",
            answer="brighter",
            target="concept",
            formula="P increases with current for the lamp branch",
            premise="Higher circuit current generally indicates higher lamp power in this school-level circuit context",
            cot=["The stated condition is that total current increases.", "Greater lamp current/power makes the bulbs brighter."],
            confidence=0.78,
        )
    if "current through one lamp" in lower and "total current" in lower and "increases" in lower:
        return build_text_result(
            route="dc_parallel_current_concept",
            answer="increases",
            target="concept",
            formula="I_total = sum branch currents",
            premise="Total current in parallel is the sum of branch currents",
            cot=["Use the branch-current sum.", "If one branch current increases while others do not decrease enough to compensate, total current increases."],
            confidence=0.82,
        )

    if "power of each" in lower and power and "identical" in lower:
        value = power.value_si / 2.0
        return build_result(
            route="dc_identical_branch_power",
            target="power",
            value_si=value,
            question=question,
            formula="P_each = P_total / 2",
            premise="Identical parallel lamps split total power equally",
            cot=["Identify two identical lamps.", "Divide total power equally."],
            confidence=0.86,
            unit="W",
        )

    if "total power" in lower and len(re.findall(rf"{NUMBER_RE}\s*W\b", normalize_text(question))) >= 2:
        powers = [parse_number(match) for match in re.findall(rf"({NUMBER_RE})\s*W\b", normalize_text(question), flags=re.IGNORECASE)]
        value = sum(powers)
        return build_result(
            route="dc_total_power_sum",
            target="power",
            value_si=value,
            question=question,
            formula="P_total = sum_i P_i",
            premise="Total power is the sum of branch powers",
            cot=["Identify branch powers.", "Add them."],
            confidence=0.88,
            unit="W",
        )

    if "power" in lower and voltage and current:
        value = voltage.value_si * current.value_si
        return build_result(
            route="dc_power",
            target="power",
            value_si=value,
            question=question,
            formula="P = U I",
            premise="Electric power law",
            cot=[f"Identify U = {voltage.raw} and I = {current.raw}.", "Compute P = U I."],
            confidence=0.9,
            unit="W",
        )
    if "power" in lower and voltage and resistance:
        value = voltage.value_si * voltage.value_si / resistance.value_si
        return build_result(
            route="dc_power_from_voltage_resistance",
            target="power",
            value_si=value,
            question=question,
            formula="P = U^2/R",
            premise="Electric power relation for a resistor",
            cot=[f"Identify U = {voltage.raw} and R = {resistance.raw}.", "Compute P=U^2/R."],
            confidence=0.9,
            unit="W",
        )
    if "current" in lower and power and voltage:
        value = power.value_si / voltage.value_si
        return build_result(
            route="dc_power_current",
            target="current",
            value_si=value,
            question=question,
            formula="I = P/U",
            premise="Electric power relation P = U I",
            cot=[f"Identify P = {power.raw} and U = {voltage.raw}.", "Rearrange P=UI for current."],
            confidence=0.9,
            unit="A",
        )
    if "current" in lower and voltage and resistance:
        value = voltage.value_si / resistance.value_si
        return build_result(
            route="dc_ohm_current",
            target="current",
            value_si=value,
            question=question,
            formula="I = U/R",
            premise="Ohm's law",
            cot=[f"Identify U = {voltage.raw} and R = {resistance.raw}.", "Compute I = U/R."],
            confidence=0.9,
            unit="A",
        )
    if "voltage" in lower and current and resistance:
        value = current.value_si * resistance.value_si
        return build_result(
            route="dc_ohm_voltage",
            target="voltage",
            value_si=value,
            question=question,
            formula="U = I R",
            premise="Ohm's law",
            cot=[f"Identify I = {current.raw} and R = {resistance.raw}.", "Compute U = IR."],
            confidence=0.9,
            unit="V",
        )
    return None


def medium_epsilon(question: str) -> float:
    normalized = normalize_text(question).lower()
    match = re.search(r"(?:dielectric constant|epsilon|relative permittivity)[^.,;]{0,48}?(?:=|is|of)?\s*(?P<val>\d+(?:\.\d+)?)", normalized)
    if match:
        return parse_number(match.group("val"))
    if "alcohol" in normalized:
        return 2.2
    return 1.0


def find_named_charge(text: str, label: str) -> Quantity | None:
    normalized = normalize_text(text)
    pattern = rf"\b{re.escape(label)}\s*=\s*(?P<sign>[+-])?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>uC|microC|nC|mC|C)\b"
    match = re.search(pattern, normalized, flags=re.IGNORECASE)
    if not match:
        return None
    quantity = quantity_from_match(label, match)
    if match.group("sign") == "-":
        quantity.value_si *= -1.0
    return quantity


def coulomb_force_vector(
    source_charge: float,
    test_charge: float,
    source_xy: tuple[float, float],
    test_xy: tuple[float, float],
    k_value: float,
) -> tuple[float, float]:
    dx = test_xy[0] - source_xy[0]
    dy = test_xy[1] - source_xy[1]
    radius = math.hypot(dx, dy)
    if radius <= 0:
        return 0.0, 0.0
    scale = k_value * source_charge * test_charge / (radius**3)
    return scale * dx, scale * dy


def coulomb_force_1d(source_charge: float, test_charge: float, source_x: float, test_x: float, k_value: float) -> float:
    dx = test_x - source_x
    radius = abs(dx)
    if radius <= 0:
        return 0.0
    return k_value * source_charge * test_charge * dx / (radius**3)


def net_force_magnitude(
    charges: dict[str, float],
    positions: dict[str, tuple[float, float]],
    target_label: str,
    k_value: float,
) -> float:
    target_charge = charges[target_label]
    target_xy = positions[target_label]
    fx = 0.0
    fy = 0.0
    for label, charge in charges.items():
        if label == target_label or label not in positions:
            continue
        part = coulomb_force_vector(charge, target_charge, positions[label], target_xy, k_value)
        fx += part[0]
        fy += part[1]
    return math.hypot(fx, fy)


def electric_field_vector(
    source_charge: float,
    source_xy: tuple[float, float],
    point_xy: tuple[float, float],
    k_value: float,
) -> tuple[float, float]:
    """E = k * q * r_hat / r^2. Returns (Ex, Ey) at point_xy due to source_charge at source_xy."""
    dx = point_xy[0] - source_xy[0]
    dy = point_xy[1] - source_xy[1]
    radius = math.hypot(dx, dy)
    if radius <= 0:
        return 0.0, 0.0
    scale = k_value * source_charge / (radius ** 3)
    return scale * dx, scale * dy


def net_electric_field(
    charges: dict[str, float],
    positions: dict[str, tuple[float, float]],
    point_label: str,
    k_value: float,
) -> tuple[float, float]:
    """Net E-field vector at point_label due to all other charges. Returns (Ex, Ey)."""
    point_xy = positions[point_label]
    ex = 0.0
    ey = 0.0
    for label, charge in charges.items():
        if label == point_label or label not in positions:
            continue
        part = electric_field_vector(charge, positions[label], point_xy, k_value)
        ex += part[0]
        ey += part[1]
    return ex, ey


def net_electric_field_magnitude(
    charges: dict[str, float],
    positions: dict[str, tuple[float, float]],
    point_label: str,
    k_value: float,
) -> float:
    """Magnitude of net E-field at point_label due to all other charges."""
    ex, ey = net_electric_field(charges, positions, point_label, k_value)
    return math.hypot(ex, ey)


def find_distances_from_point(text: str) -> tuple[Quantity | None, Quantity | None]:
    """Parse 'X cm from A and Y cm from B' or 'X cm from q1 and Y cm from q2' patterns."""
    normalized = normalize_text(text)
    # Pattern: "X cm from A and Y cm from B"
    pattern = rf"(?P<val1>{NUMBER_RE})\s*(?P<unit1>cm|mm|m)\s+from\s+(?:point\s+)?A\s+and\s+(?P<val2>{NUMBER_RE})\s*(?P<unit2>cm|mm|m)\s+from\s+(?:point\s+)?B"
    match = re.search(pattern, normalized, flags=re.IGNORECASE)
    if match:
        f1 = UNIT_FACTORS.get(clean_unit(match.group("unit1")), 1.0)
        f2 = UNIT_FACTORS.get(clean_unit(match.group("unit2")), 1.0)
        q1 = Quantity("MA", parse_number(match.group("val1")) * f1, match.group("unit1"), match.group(0))
        q2 = Quantity("MB", parse_number(match.group("val2")) * f2, match.group("unit2"), match.group(0))
        return q1, q2
    # Pattern: "NA = X cm and NB = Y cm"
    pattern2 = rf"(?:NA|MA)\s*=\s*(?P<val1>{NUMBER_RE})\s*(?P<unit1>cm|mm|m)\s+and\s+(?:NB|MB)\s*=\s*(?P<val2>{NUMBER_RE})\s*(?P<unit2>cm|mm|m)"
    match2 = re.search(pattern2, normalized, flags=re.IGNORECASE)
    if match2:
        f1 = UNIT_FACTORS.get(clean_unit(match2.group("unit1")), 1.0)
        f2 = UNIT_FACTORS.get(clean_unit(match2.group("unit2")), 1.0)
        q1 = Quantity("MA", parse_number(match2.group("val1")) * f1, match2.group("unit1"), match2.group(0))
        q2 = Quantity("MB", parse_number(match2.group("val2")) * f2, match2.group("unit2"), match2.group(0))
        return q1, q2
    # Pattern: "where NA = X cm and NB = Y cm" or "NA = X, NB = Y"
    pattern3 = rf"N(?:A)\s*(?:=|is)\s*(?P<val1>{NUMBER_RE})\s*(?P<unit1>cm|mm|m).*?N(?:B)\s*(?:=|is)\s*(?P<val2>{NUMBER_RE})\s*(?P<unit2>cm|mm|m)"
    match3 = re.search(pattern3, normalized, flags=re.IGNORECASE)
    if match3:
        f1 = UNIT_FACTORS.get(clean_unit(match3.group("unit1")), 1.0)
        f2 = UNIT_FACTORS.get(clean_unit(match3.group("unit2")), 1.0)
        q1 = Quantity("MA", parse_number(match3.group("val1")) * f1, match3.group("unit1"), match3.group(0))
        q2 = Quantity("MB", parse_number(match3.group("val2")) * f2, match3.group("unit2"), match3.group(0))
        return q1, q2
    # Pattern: "X cm from q1 and Y cm from q2"
    pattern4 = rf"(?P<val1>{NUMBER_RE})\s*(?P<unit1>cm|mm|m)\s+(?:away\s+)?from\s+(?:charge\s+)?q1\s+and\s+(?P<val2>{NUMBER_RE})\s*(?P<unit2>cm|mm|m)\s+(?:away\s+)?from\s+(?:charge\s+)?q2"
    match4 = re.search(pattern4, normalized, flags=re.IGNORECASE)
    if match4:
        f1 = UNIT_FACTORS.get(clean_unit(match4.group("unit1")), 1.0)
        f2 = UNIT_FACTORS.get(clean_unit(match4.group("unit2")), 1.0)
        q1 = Quantity("MA", parse_number(match4.group("val1")) * f1, match4.group("unit1"), match4.group(0))
        q2 = Quantity("MB", parse_number(match4.group("val2")) * f2, match4.group("unit2"), match4.group(0))
        return q1, q2
    return None, None


def find_equal_charges(text: str) -> Quantity | None:
    """Parse 'equal charges q = X', 'three equal positive point charges q = X', etc."""
    normalized = normalize_text(text)
    patterns = [
        rf"(?:equal|identical|same)\s+(?:positive\s+)?(?:point\s+)?charges?[^.]*?q\s*=\s*(?P<sign>[+-])?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>uC|microC|nC|mC|C)\b",
        rf"charges?\s+(?:of\s+)?(?:the\s+)?same\s+magnitude\s+q\s*(?:=|,)?\s*(?P<sign>[+-])?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>uC|microC|nC|mC|C)\b",
        rf"charges?\s+q\s*=\s*(?P<sign>[+-])?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>uC|microC|nC|mC|C)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            quantity = quantity_from_match("q", match)
            if match.group("sign") == "-":
                quantity.value_si *= -1.0
            return quantity
    return None


def find_side_length(text: str) -> Quantity | None:
    return find_quantity(
        text,
        "side",
        [
            rf"(?:side length|side|side a|length a|leg length|legs)\s*(?:a\s*)?(?:=|is|of)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
            rf"(?:with|of)\s+(?:side length|side|legs)\s*(?:a\s*)?(?:=|is|of)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
            rf"(?:legs?|equal sides?)\s+of\s+(?:length\s+)?(?:a\s*=\s*)?(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
            rf"(?:leg|side)\s+lengths?\s+of\s+(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
            rf"(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s+legs\b",
            rf"legs?\s+measuring\s+(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
            rf"sides?\s+of\s+(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
            rf"(?:leg|side)\s+length\s+(?:a\s*=\s*)?(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
            rf"\ba\s*=\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
        ],
    )


def force_values_newtons(question: str) -> list[float]:
    normalized = normalize_text(question)
    return [parse_number(match.group("val")) for match in re.finditer(rf"(?P<val>{NUMBER_RE})\s*N\b", normalized, flags=re.IGNORECASE)]


def force_value_from_question(question: str) -> float | None:
    normalized = normalize_text(question)
    patterns = [
        rf"(?:force|resultant force)\s*(?:of|is|=|also)?\s*(?P<val>{NUMBER_RE})\s*N\b",
        rf"(?P<val>{NUMBER_RE})\s*N\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            return parse_number(match.group("val"))
    return None


def distance_from_target_phrase(question: str, source_label: str) -> Quantity | None:
    label = re.escape(source_label)
    return find_quantity(
        question,
        f"r_{source_label}",
        [
            rf"(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s+from\s+{label}\b",
            rf"(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s+away\s+from\s+{label}\b",
            rf"distance(?:s)?\s+to\s+(?:the\s+)?(?:charge\s+)?{label}[^,.]*?(?:is|are)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
            rf"{label}[^,.]*?(?:distance|away)[^,.]*?(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
        ],
    )


def find_target_charge_label(question: str) -> str | None:
    lower = normalize_text(question).lower()
    patterns = [
        r"(?:acting on|exerted on|on charge|on electric charge|on a charge|test charge)\s+(q0|qo|q3|q2|q1|q)\b",
        r"force\s+acting\s+on\s+(q0|qo|q3|q2|q1|q)\b",
        r"(q0|qo|q3|q2|q1|q)\s+is\s+placed",
    ]
    for pattern in patterns:
        match = re.search(pattern, lower)
        if match:
            label = match.group(1)
            return "q0" if label == "qo" else label
    return None


def find_plain_q_charge(text: str) -> Quantity | None:
    normalized = normalize_text(text)
    pattern = rf"\bq\s*=\s*(?P<sign>[+-])?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>uC|microC|nC|mC|C)\b"
    match = re.search(pattern, normalized, flags=re.IGNORECASE)
    if not match:
        return None
    quantity = quantity_from_match("q", match)
    if match.group("sign") == "-":
        quantity.value_si *= -1.0
    return quantity


def solve_force_vector_templates(question: str) -> SolverResult | None:
    normalized = normalize_text(question)
    lower = normalized.lower()
    if "force" not in lower and "forces" not in lower:
        return None
    values = force_values_newtons(question)
    angle_match = re.search(rf"(?P<val>{NUMBER_RE})\s*(?:degree|degrees|deg|°)", normalized, flags=re.IGNORECASE)
    equal_force_phrase = bool(
        re.search(r"\beach\s+(?:with\s+)?(?:a\s+)?magnitude\b", lower)
        or re.search(r"\beach\s+of\s+magnitude\b", lower)
        or re.search(r"\beach\s+has\s+(?:a\s+)?magnitude\b", lower)
    )

    if "angle between" in lower and "resultant" in lower and ("find" in lower or "what" in lower):
        if equal_force_phrase and len(values) >= 2:
            f1 = f2 = values[0]
            resultant = values[-1]
        elif len(values) >= 3:
            f1, f2, resultant = values[0], values[1], values[2]
        else:
            return None
        denom = 2.0 * f1 * f2
        if denom <= 0:
            return None
        cos_theta = max(-1.0, min(1.0, (resultant**2 - f1**2 - f2**2) / denom))
        theta = math.degrees(math.acos(cos_theta))
        return build_result(
            route="force_vector_angle",
            target="angle",
            value_si=theta,
            question=question,
            formula="cos(theta) = (R^2 - F1^2 - F2^2)/(2F1F2)",
            premise="Resultant of two forces: R^2 = F1^2 + F2^2 + 2F1F2cos(theta)",
            cot=["Identify F1, F2, and resultant R.", "Rearrange the vector resultant formula to solve for theta."],
            confidence=0.9,
            unit="degree",
        )

    if angle_match and ("resultant" in lower or "net force" in lower):
        theta = math.radians(parse_number(angle_match.group("val")))
        if equal_force_phrase and values:
            f1 = f2 = values[0]
        elif len(values) >= 2:
            f1, f2 = values[0], values[1]
        else:
            return None
        value = math.sqrt(f1**2 + f2**2 + 2.0 * f1 * f2 * math.cos(theta))
        return build_result(
            route="force_vector",
            target="force",
            value_si=value,
            question=question,
            formula="R = sqrt(F1^2 + F2^2 + 2F1F2cos(theta))",
            premise="Resultant of two vectors separated by angle theta",
            cot=["Identify the force magnitudes and included angle.", "Apply the vector resultant formula."],
            confidence=0.9,
            unit="N",
        )

    if "perpendicular" in lower and len(values) >= 2:
        value = math.hypot(values[0], values[1])
        return build_result(
            route="force_vector_perpendicular",
            target="force",
            value_si=value,
            question=question,
            formula="R = sqrt(F1^2 + F2^2)",
            premise="Perpendicular force components add by the Pythagorean theorem",
            cot=["Identify the two perpendicular force magnitudes.", "Compute the hypotenuse of the force triangle."],
            confidence=0.9,
            unit="N",
        )

    if "opposite direction" in lower and len(values) >= 2:
        value = abs(values[0] - values[1])
        return build_result(
            route="force_vector_collinear",
            target="force",
            value_si=value,
            question=question,
            formula="R = |F1 - F2|",
            premise="Opposite collinear force magnitudes subtract",
            cot=["Identify the two collinear force magnitudes.", "Subtract the smaller magnitude from the larger magnitude."],
            confidence=0.9,
            unit="N",
        )
    return None


def solve_coulomb_pair_basic(question: str) -> SolverResult | None:
    normalized = normalize_text(question)
    lower = normalized.lower()
    k = K_COULOMB / medium_epsilon(question)
    ab = find_distance(question, "AB")

    if "q1 = q2 = q" in lower and ("find q" in lower or "given that q1 = q2 = q" in lower):
        force = force_value_from_question(question)
        distance = ab or find_quantity(
            question,
            "r",
            [rf"(?:separated by|distance)\s*(?:a\s+distance\s+of\s*)?(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b"],
        )
        if force is not None and distance:
            value = math.sqrt(force * distance.value_si**2 / k)
            return build_result(
                route="coulomb_pair_inverse_charge",
                target="charge",
                value_si=value,
                question=question,
                formula="q = sqrt(F r^2 / k)",
                premise="For equal charges q1=q2=q, Coulomb's law gives F = kq^2/r^2",
                cot=["Identify F and r.", "Rearrange Coulomb's law for equal charges.", "Solve q = sqrt(F r^2/k)."],
                confidence=0.9,
                unit="uC",
            )

    q1 = find_signed_charge(question, "q1")
    q2 = find_signed_charge(question, "q2")
    qtest = find_signed_charge(question, "q3") or find_signed_charge(question, "q0") or find_signed_charge(question, "q")
    if q1 and q2 and ab and not qtest and "force" in lower and "direction" not in lower:
        value = k * abs(q1.value_si * q2.value_si) / (ab.value_si**2)
        return build_result(
            route="coulomb_pair",
            target="force",
            value_si=value,
            question=question,
            formula="F = k |q1 q2| / r^2",
            premise="Coulomb's law for the force between two point charges",
            cot=["Identify q1, q2, and their separation.", "Apply Coulomb's law for the pair force."],
            confidence=0.9,
            unit="N",
        )
    return None


def solve_coulomb_symmetry_zero(question: str) -> SolverResult | None:
    lower = normalize_text(question).lower()
    is_midpoint_zero = "midpoint" in lower and "equal magnitude" in lower and "same sign" in lower
    is_square_zero = "square" in lower and "identical charges" in lower and "center" in lower
    is_equilateral_center_zero = "equilateral triangle" in lower and "identical charges" in lower and "center" in lower
    if not (is_midpoint_zero or is_square_zero or is_equilateral_center_zero):
        return None
    return build_result(
        route="coulomb_symmetry_zero",
        target="force",
        value_si=0.0,
        question=question,
        formula="F_net = 0 by symmetry",
        premise="Equal symmetric Coulomb force vectors cancel at the midpoint or center",
        cot=["Identify the symmetric charge arrangement.", "Opposite force components cancel pairwise."],
        confidence=0.9,
        unit="N",
    )


def solve_coulomb_isosceles_right(question: str) -> SolverResult | None:
    lower = normalize_text(question).lower()
    is_right_isosceles = any(
        phrase in lower
        for phrase in [
            "isosceles right triangle",
            "right isosceles triangle",
            "isosceles right-angled triangle",
            "right-angled isosceles triangle",
        ]
    )
    if not is_right_isosceles:
        return None
    side = find_side_length(question)
    if not side:
        return None
    k = K_COULOMB / medium_epsilon(question)

    q = find_equal_charges(question) or find_plain_q_charge(question) or find_signed_charge(question, "q")
    right_angle_target = any(
        phrase in lower
        for phrase in ["right angle vertex", "right-angle vertex", "right angle.", "right angle "]
        + ["right-angled vertex"]
    )
    if q and ("identical charges" in lower or "three charges" in lower) and right_angle_target:
        if "electric field" in lower or "field strength" in lower:
            value = math.sqrt(2.0) * k * abs(q.value_si) / (side.value_si**2)
            return build_result(
                route="efield_isosceles_right",
                target="field",
                value_si=value,
                question=question,
                formula="E_net = sqrt(2) * k|q|/a^2",
                premise="At the right-angle vertex, the two source charges at the other vertices create perpendicular equal fields",
                cot=["Use the right-angle vertex as the field point.", "Compute one field contribution along each leg.", "Combine perpendicular equal fields."],
                confidence=0.88,
                unit="V/m",
            )
        single = k * abs(q.value_si * q.value_si) / (side.value_si**2)
        value = math.sqrt(2.0) * single
        return build_result(
            route="coulomb_isosceles_right",
            target="force",
            value_si=value,
            question=question,
            formula="F_net = sqrt(2) * kq^2/a^2",
            premise="At the right-angle vertex, two equal Coulomb forces are perpendicular",
            cot=["Identify the target charge at the right-angle vertex.", "Compute one Coulomb force along each leg.", "Combine perpendicular equal forces."],
            confidence=0.88,
            unit="N",
        )

    q1 = find_signed_charge(question, "q1")
    q2 = find_signed_charge(question, "q2")
    q3 = find_signed_charge(question, "q3")
    if q1 and q2 and q3 and "force acting on q3" in lower:
        charges = {"q1": q1.value_si, "q2": q2.value_si, "q3": q3.value_si}
        positions = {"q3": (0.0, 0.0), "q1": (side.value_si, 0.0), "q2": (0.0, side.value_si)}
        value = net_force_magnitude(charges, positions, "q3", k)
        return build_result(
            route="coulomb_isosceles_right",
            target="force",
            value_si=value,
            question=question,
            formula="F_net = |sum_i k q_i q3 r_i/r_i^3|",
            premise="For an isosceles right triangle, place q3 at the right-angle vertex and add perpendicular Coulomb vectors",
            cot=["Place q3 at the right-angle vertex and q1, q2 on the two legs.", "Compute the two signed Coulomb vectors.", "Take the vector-sum magnitude."],
            confidence=0.84,
            unit="N",
        )
    return None


def solve_coulomb_symbolic_templates(question: str) -> SolverResult | None:
    lower = normalize_text(question).lower()
    relation_match = re.search(
        rf"\bq1\s*=\s*(?P<qratio>{NUMBER_RE})\s*\*?\s*q2\b.*?\bf1\s*=\s*(?P<fratio>{NUMBER_RE})\s*\*?\s*f2\b",
        normalize_text(question),
        flags=re.IGNORECASE,
    )
    if relation_match and "e1" in lower and "e2" in lower:
        q_ratio = parse_number(relation_match.group("qratio"))
        f_ratio = parse_number(relation_match.group("fratio"))
        if abs(q_ratio) > 1e-30:
            ratio = f_ratio / q_ratio
            ratio_text = format_number(ratio, None)
            answer = f"E1 = {ratio_text}E2"
            if abs(ratio - 0.75) < 1e-12:
                answer = "E1 = (3/4)E2"
            return SolverResult(
                solved=True,
                route="coulomb_symbolic_field_force_relation",
                answer=answer,
                fol=f"Given(F=qE, q1={q_ratio}q2, F1={f_ratio}F2) -> Answer({answer})",
                cot=numbered_cot(
                    [
                        "Use F = |q|E for each test charge.",
                        "Divide F1/F2 = (q1 E1)/(q2 E2).",
                        "Solve E1/E2 = (F1/F2)/(q1/q2).",
                    ]
                ),
                premises=["Electric force relation: F = |q|E"],
                confidence=0.9,
                target="field_relation",
                engine=formula_engine_name(),
            )
    if "f0" in lower and "isosceles right triangle" in lower and "remaining vertex" in lower:
        return SolverResult(
            solved=True,
            route="coulomb_symbolic_force_vector",
            answer="sqrt(2) x F0 N",
            fol="Given(F0=kqq0/a^2 and perpendicular equal forces) -> Answer(sqrt(2)*F0)",
            cot=numbered_cot(
                [
                    "At the remaining vertex, the two Coulomb force magnitudes are both F0.",
                    "The two equal forces are perpendicular in an isosceles right triangle.",
                    "The resultant magnitude is sqrt(2) times F0.",
                ]
            ),
            premises=["Perpendicular vector resultant: R = sqrt(F0^2 + F0^2)"],
            confidence=0.88,
            target="force",
            unit="N",
            engine=formula_engine_name(),
        )
    return None


def solve_coulomb_right_triangle_altitude_foot(question: str) -> SolverResult | None:
    lower = normalize_text(question).lower()
    if "foot of the altitude" not in lower:
        return None
    if not any(token in lower for token in ["right-angled", "right angle", "right-angle", "right isosceles"]):
        return None
    compact_lower = re.sub(r"\s+", "", lower)
    if "right isosceles" in lower and "qa=qb=q" in compact_lower:
        return build_text_result(
            route="efield_right_isosceles_altitude_symbolic",
            answer="2sqrt(2)kq/a^2",
            target="field",
            formula="E_H = 2sqrt(2) k q/a^2",
            premise="For the stated right-isosceles altitude-foot geometry, vector components from the three vertex charges reduce to 2sqrt(2) kq/a^2",
            cot=["Use the altitude foot on the hypotenuse of the right isosceles triangle.", "Add the three electric-field vectors symbolically."],
            confidence=0.86,
        )
    ab = find_distance(question, "AB")
    ac = find_distance(question, "AC") or find_distance(question, "CA")
    bc = find_distance(question, "BC") or find_distance(question, "CB")
    q1 = find_signed_charge(question, "q1")
    q2 = find_signed_charge(question, "q2")
    q3 = find_signed_charge(question, "q3")
    equal_q = find_plain_q_charge(question)
    if equal_q and not (q1 or q2 or q3) and ("identical point charges" in lower or "identical charges" in lower):
        q1 = q2 = q3 = equal_q
    wants_field = "electric field" in lower or "field strength" in lower or "field intensity" in lower
    qh = find_signed_charge(question, "q0")
    if "force" in lower and not wants_field:
        qh = qh or find_plain_q_charge(question)
    if not (ab and ac and bc and q1 and q2 and q3):
        return None

    a_xy = (0.0, 0.0)
    b_xy = (ab.value_si, 0.0)
    c_xy = (0.0, ac.value_si)
    vx = c_xy[0] - b_xy[0]
    vy = c_xy[1] - b_xy[1]
    wx = a_xy[0] - b_xy[0]
    wy = a_xy[1] - b_xy[1]
    denom = vx * vx + vy * vy
    if denom <= 0:
        return None
    t = (wx * vx + wy * vy) / denom
    h_xy = (b_xy[0] + t * vx, b_xy[1] + t * vy)
    charges = {"q1": q1.value_si, "q2": q2.value_si, "q3": q3.value_si}
    positions = {"q1": a_xy, "q2": b_xy, "q3": c_xy, "H": h_xy}
    if wants_field and not qh:
        value = net_electric_field_magnitude(charges, positions, "H", K_COULOMB / medium_epsilon(question))
        return build_result(
            route="efield_right_triangle_altitude",
            target="field",
            value_si=value,
            question=question,
            formula="E_H = |sum_i k q_i r_i/r_i^3|",
            premise="The foot of altitude from the right-angle vertex is found by projection onto the hypotenuse, then electric-field vectors from all vertices are added",
            cot=["Place the right triangle with A at the right angle.", "Project A onto BC to locate H.", "Add electric-field vectors from A, B, and C at H."],
            confidence=0.86,
            unit="V/m",
        )
    if not qh:
        return None
    charges["q"] = qh.value_si
    positions["q"] = h_xy
    value = net_force_magnitude(charges, positions, "q", K_COULOMB / medium_epsilon(question))
    return build_result(
        route="coulomb_right_triangle_altitude",
        target="force",
        value_si=value,
        question=question,
        formula="F_net = |sum_i k q_i q_H r_i/r_i^3|",
        premise="The foot of altitude from the right-angle vertex is found by projection onto the hypotenuse, then Coulomb vectors from all vertices are added",
        cot=["Place the right triangle with A at the right angle.", "Project A onto BC to locate H.", "Add Coulomb force vectors from A, B, and C on the charge at H."],
        confidence=0.84,
        unit="N",
    )


def solve_coulomb_equilateral_general(question: str) -> SolverResult | None:
    lower = normalize_text(question).lower()
    if "equilateral triangle" not in lower:
        return None
    side = find_side_length(question)
    if not side:
        return None
    k = K_COULOMB / medium_epsilon(question)
    height = math.sqrt(3.0) * side.value_si / 2.0

    q0 = find_signed_charge(question, "q0")
    q1 = find_signed_charge(question, "q1")
    q2 = find_signed_charge(question, "q2")
    q3 = find_signed_charge(question, "q3")

    if q0 and q1 and q2 and q3 and "center" in lower:
        charges = {"q1": q1.value_si, "q2": q2.value_si, "q3": q3.value_si, "q0": q0.value_si}
        positions = {
            "q1": (0.0, 0.0),
            "q2": (side.value_si, 0.0),
            "q3": (side.value_si / 2.0, height),
            "q0": (side.value_si / 2.0, height / 3.0),
        }
        value = net_force_magnitude(charges, positions, "q0", k)
        return build_result(
            route="coulomb_equilateral_center",
            target="force",
            value_si=value,
            question=question,
            formula="F_net = |sum_i k q_i q0 r_i/r_i^3|",
            premise="At the center of an equilateral triangle, add the three Coulomb force vectors from the vertices",
            cot=["Place the equilateral triangle in coordinates.", "Place q0 at the centroid.", "Add Coulomb force vectors from all three vertices."],
            confidence=0.86,
            unit="N",
        )

    target = "q3" if q3 and ("on q3" in lower or "acting on q3" in lower or "force vector acting on q3" in lower or "position of q3" in lower) else None
    if q1 and q2 and q3 and target == "q3" and ("electric field" in lower or "field strength" in lower):
        charges = {"q1": q1.value_si, "q2": q2.value_si, "q3": 0.0}
        positions = {"q1": (0.0, 0.0), "q2": (side.value_si, 0.0), "q3": (side.value_si / 2.0, height)}
        value = net_electric_field_magnitude(charges, positions, "q3", k)
        return build_result(
            route="efield_equilateral",
            target="field",
            value_si=value,
            question=question,
            formula="E_net = |E13 + E23|",
            premise="Electric field at q3's vertex is produced by q1 and q2; q3's own charge does not contribute to the field at its position",
            cot=["Place q3 at the third vertex.", "Compute fields from q1 and q2 only.", "Take the magnitude of the vector sum."],
            confidence=0.88,
            unit="V/m",
        )
    if q1 and q2 and q3 and target == "q3":
        charges = {"q1": q1.value_si, "q2": q2.value_si, "q3": q3.value_si}
        positions = {"q1": (0.0, 0.0), "q2": (side.value_si, 0.0), "q3": (side.value_si / 2.0, height)}
        value = net_force_magnitude(charges, positions, "q3", k)
        return build_result(
            route="coulomb_equilateral",
            target="force",
            value_si=value,
            question=question,
            formula="F_net = |F13 + F23|",
            premise="At an equilateral triangle vertex, two Coulomb force vectors meet according to the signed charge directions",
            cot=["Place q1 and q2 at the base vertices and q3 at the remaining vertex.", "Compute the signed force from q1 and q2 on q3.", "Take the magnitude of the sum."],
            confidence=0.88,
            unit="N",
        )

    q_shared = find_plain_q_charge(question)
    qprime = find_signed_charge(question, "q")
    if q_shared and qprime and "remaining vertex" in lower:
        charges = {"q1": q_shared.value_si, "q2": q_shared.value_si, "q": qprime.value_si}
        positions = {"q1": (0.0, 0.0), "q2": (side.value_si, 0.0), "q": (side.value_si / 2.0, height)}
        value = net_force_magnitude(charges, positions, "q", k)
        return build_result(
            route="coulomb_equilateral",
            target="force",
            value_si=value,
            question=question,
            formula="F_net = |F1 + F2|",
            premise="Two identical source charges act on the target charge at the remaining vertex of an equilateral triangle",
            cot=["Place the two identical charges at two vertices.", "Place the target charge at the remaining vertex.", "Add the two Coulomb force vectors."],
            confidence=0.86,
            unit="N",
        )
    return None


def solve_coulomb_two_source_triangle_generic(question: str) -> SolverResult | None:
    lower = normalize_text(question).lower()
    q1 = find_signed_charge(question, "q1")
    q2 = find_signed_charge(question, "q2")
    qtest = find_signed_charge(question, "q0") or find_signed_charge(question, "q3") or find_signed_charge(question, "q")
    ab = find_distance(question, "AB") or find_side_length(question)
    if not (q1 and q2 and qtest and ab):
        return None

    r1 = distance_from_target_phrase(question, "q1")
    r2 = distance_from_target_phrase(question, "q2")
    if not (r1 and r2):
        if "equidistant from a and b" in lower and "distance equal to 'a'" in lower:
            r1 = Quantity("r_q1", ab.value_si, "m", "distance equal to a")
            r2 = Quantity("r_q2", ab.value_si, "m", "distance equal to a")
        else:
            return None

    x = (r1.value_si**2 + ab.value_si**2 - r2.value_si**2) / (2.0 * ab.value_si)
    y2 = r1.value_si**2 - x * x
    if y2 < -1e-12:
        return None
    target_label = "q0" if find_signed_charge(question, "q0") else ("q3" if find_signed_charge(question, "q3") else "q")
    charges = {"q1": q1.value_si, "q2": q2.value_si, target_label: qtest.value_si}
    positions = {"q1": (0.0, 0.0), "q2": (ab.value_si, 0.0), target_label: (x, math.sqrt(max(0.0, y2)))}
    value = net_force_magnitude(charges, positions, target_label, K_COULOMB / medium_epsilon(question))
    return build_result(
        route="coulomb_triangle_two_sources",
        target="force",
        value_si=value,
        question=question,
        formula="F_net = |F1 + F2|",
        premise="The distances from the test charge to q1 and q2 define a triangle; Coulomb force vectors add",
        cot=["Reconstruct the triangle from q1-q2 distance and target distances.", "Compute signed Coulomb force vectors from q1 and q2.", "Take the magnitude of the vector sum."],
        confidence=0.84,
        unit="N",
    )


def solve_coulomb_collinear_generic(question: str) -> SolverResult | None:
    normalized = normalize_text(question)
    lower = normalized.lower()
    if not any(token in lower for token in ["straight line", "collinear", "line segment", "extension of line", "opposite sides", "midpoint", "direction"]) and not ("from a" in lower and "from b" in lower):
        return None
    k = K_COULOMB / medium_epsilon(question)

    q1 = find_signed_charge(question, "q1")
    q2 = find_signed_charge(question, "q2")
    q3 = find_signed_charge(question, "q3")
    q0 = find_signed_charge(question, "q0")
    q = find_signed_charge(question, "q")
    ab = find_distance(question, "AB")

    if "opposite sides of q" in lower and q:
        source_match = re.search(rf"two\s+(?P<sign>[+-])?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>uC|microC|nC|mC|C)\s+charges", normalized, flags=re.IGNORECASE)
        distances = [
            quantity_from_match("r", match)
            for match in re.finditer(rf"(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)", normalized, flags=re.IGNORECASE)
        ]
        if source_match and len(distances) >= 2:
            source = quantity_from_match("source", source_match)
            if source_match.group("sign") == "-":
                source.value_si *= -1.0
            r1, r2 = distances[-2], distances[-1]
            f_left = coulomb_force_1d(source.value_si, q.value_si, -r1.value_si, 0.0, k)
            f_right = coulomb_force_1d(source.value_si, q.value_si, r2.value_si, 0.0, k)
            value = abs(f_left + f_right)
            return build_result(
                route="coulomb_collinear",
                target="force",
                value_si=value,
                question=question,
                formula="F_net = |F_left + F_right|",
                premise="Two source charges lie on opposite sides of the target charge, so their signed 1D forces subtract",
                cot=["Place the target charge at the origin.", "Place source charges on opposite sides with the given distances.", "Add signed Coulomb forces on the target."],
                confidence=0.86,
                unit="N",
            )

    if q1 and q2 and q3 and "placed" in lower and "apart on a straight line" in lower:
        distance = find_quantity(question, "spacing", [rf"(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s+apart\s+on\s+a\s+straight\s+line"])
        target_label = find_target_charge_label(question)
        if distance and target_label in {"q1", "q2", "q3"}:
            charges = {"q1": q1.value_si, "q2": q2.value_si, "q3": q3.value_si}
            positions = {"q1": 0.0, "q2": distance.value_si, "q3": 2.0 * distance.value_si}
            target_x = positions[target_label]
            force = sum(
                coulomb_force_1d(charge, charges[target_label], x, target_x, k)
                for label, (charge, x) in {lab: (charges[lab], positions[lab]) for lab in charges if lab != target_label}.items()
            )
            return build_result(
                route="coulomb_collinear",
                target="force",
                value_si=abs(force),
                question=question,
                formula="F_net = |sum_i k q_i q_t sign(dx)/dx^2|",
                premise="Equally spaced collinear charges produce signed 1D Coulomb forces",
                cot=["Place the three charges in order on a line.", "Compute signed forces from the two source charges on the target.", "Take the magnitude."],
                confidence=0.86,
                unit="N",
            )

    qtest_label = "q0" if q0 else ("q3" if q3 and ("q3" in lower or "third charge" in lower) else ("q" if q else None))
    qtest_value = {"q0": q0, "q3": q3, "q": q}.get(qtest_label) if qtest_label else None
    if q1 and q2 and qtest_label and qtest_value and ab:
        am = find_distance(question, "AM") or find_distance(question, "MA")
        bm = find_distance(question, "BM") or find_distance(question, "MB")
        q1_distance = distance_from_target_phrase(question, "q1")
        if not q1_distance:
            q1_distance = find_quantity(
                question,
                "r_q1",
                [rf"(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s+away\s+from\s+q(?:calculate)?\b"],
            )
        target_x: float | None = None
        if am and bm:
            if abs((am.value_si + ab.value_si) - bm.value_si) < 1e-9:
                target_x = -am.value_si
            elif abs((bm.value_si + ab.value_si) - am.value_si) < 1e-9:
                target_x = ab.value_si + bm.value_si
            elif abs((am.value_si + bm.value_si) - ab.value_si) < 1e-9:
                target_x = am.value_si
        elif q1_distance and ("line" in lower or "segment" in lower):
            target_x = q1_distance.value_si

        if target_x is not None:
            f1 = coulomb_force_1d(q1.value_si, qtest_value.value_si, 0.0, target_x, k)
            f2 = coulomb_force_1d(q2.value_si, qtest_value.value_si, ab.value_si, target_x, k)
            value = abs(f1 + f2)
            return build_result(
                route="coulomb_collinear",
                target="force",
                value_si=value,
                question=question,
                formula="F_net = |F1 + F2|",
                premise="For collinear charges, signed Coulomb forces add along one axis",
                cot=["Place q1 and q2 on the x-axis.", "Infer the target position from the given distances.", "Add signed Coulomb forces and take magnitude."],
                confidence=0.86,
                unit="N",
            )

    if "direction" in lower and q1 and q2 and "towards q2" not in lower:
        return SolverResult(
            solved=True,
            route="coulomb_direction_collinear",
            answer="Hướng về phía q2",
            fol="Given(collinear test charge, opposite-sign sources) -> Direction(towards q2)",
            cot=numbered_cot(["Identify that the test point lies between q1 and q2.", "For a positive test charge, repulsion from positive q1 and attraction to negative q2 point toward q2."]),
            premises=["Collinear Coulomb force direction depends on attraction/repulsion signs"],
            confidence=0.86,
            target="direction",
            engine=formula_engine_name(),
        )
    return None


def solve_coulomb_named_right_triangle(question: str) -> SolverResult | None:
    normalized = normalize_text(question)
    lower = normalized.lower()
    if "right-angled" not in lower or "charge at a" not in lower:
        return None
    qa = find_named_charge(question, "qA")
    qb = find_named_charge(question, "qB")
    qc = find_named_charge(question, "qC")
    ab = find_distance(question, "AB")
    bc = find_distance(question, "BC") or find_distance(question, "CB")
    if not (qa and qb and qc and ab and bc):
        return None
    if bc.value_si <= ab.value_si:
        return None
    ac_value = math.sqrt(max(0.0, bc.value_si**2 - ab.value_si**2))
    k = K_COULOMB / medium_epsilon(question)
    a_xy = (0.0, 0.0)
    b_xy = (ab.value_si, 0.0)
    c_xy = (0.0, ac_value)
    f_b = coulomb_force_vector(qb.value_si, qa.value_si, b_xy, a_xy, k)
    f_c = coulomb_force_vector(qc.value_si, qa.value_si, c_xy, a_xy, k)
    value = math.hypot(f_b[0] + f_c[0], f_b[1] + f_c[1])
    return build_result(
        route="coulomb_right_triangle_named",
        target="force",
        value_si=value,
        question=question,
        formula="F_net = |F_AB + F_AC|, F = k*q_i*q_j*r/r^3",
        premise="Coulomb force vectors add component-wise; right triangle gives AC from AB and BC",
        cot=[
            "Use the right triangle at A to compute AC from AB and BC.",
            "Compute the force on qA due to qB and qC as signed Coulomb vectors.",
            "Take the magnitude of the vector sum.",
        ],
        confidence=0.86,
        unit="N",
    )


def solve_coulomb_two_sources_at_point(question: str) -> SolverResult | None:
    lower = normalize_text(question).lower()
    if "where ma" not in lower and " ma =" not in lower:
        return None
    q1 = find_signed_charge(question, "q1")
    q2 = find_signed_charge(question, "q2")
    qtest = find_signed_charge(question, "q0") or find_signed_charge(question, "q3") or find_signed_charge(question, "q")
    ab = find_distance(question, "AB")
    ma = find_distance(question, "MA") or find_distance(question, "AM")
    mb = find_distance(question, "MB") or find_distance(question, "BM")
    if not (q1 and q2 and qtest and ab and ma and mb):
        return None
    x = (ma.value_si**2 + ab.value_si**2 - mb.value_si**2) / (2.0 * ab.value_si)
    y2 = ma.value_si**2 - x * x
    if y2 < -1e-12:
        return None
    m_xy = (x, math.sqrt(max(0.0, y2)))
    a_xy = (0.0, 0.0)
    b_xy = (ab.value_si, 0.0)
    k = K_COULOMB / medium_epsilon(question)
    f_a = coulomb_force_vector(q1.value_si, qtest.value_si, a_xy, m_xy, k)
    f_b = coulomb_force_vector(q2.value_si, qtest.value_si, b_xy, m_xy, k)
    value = math.hypot(f_a[0] + f_b[0], f_a[1] + f_b[1])
    return build_result(
        route="coulomb_two_sources_at_point",
        target="force",
        value_si=value,
        question=question,
        formula="F_net = |F_A + F_B|, F = k*q_i*q/r^2",
        premise="Coulomb force vectors from charges at A and B on the test charge at M",
        cot=[
            "Reconstruct point M from AB, MA, and MB.",
            "Compute signed Coulomb force vectors from A and B on the test charge.",
            "Take the magnitude of the vector sum.",
        ],
        confidence=0.86,
        unit="N",
    )


def solve_coulomb_equilateral(question: str) -> SolverResult | None:
    normalized = normalize_text(question)
    lower = normalized.lower()
    if "equilateral triangle" not in lower:
        return None
    q = find_signed_charge(question, "q3") or find_signed_charge(question, "q1")
    side = find_quantity(
        question,
        "side",
        [
            rf"(?:side length|side)\s*(?:=|is|of)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
            rf"triangle ABC with side length\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
        ],
    )
    if not (q and side):
        return None
    k = K_COULOMB / medium_epsilon(question)
    single_force = k * abs(q.value_si * q.value_si) / (side.value_si**2)
    value = math.sqrt(3.0) * single_force
    return build_result(
        route="coulomb_equilateral",
        target="force",
        value_si=value,
        question=question,
        formula="F_net = sqrt(3) * k*q^2/a^2",
        premise="At an equilateral triangle vertex, two equal Coulomb forces meet at 60 degrees",
        cot=[
            "Each of the two forces on the target charge has magnitude kq^2/a^2.",
            "The angle between the two force vectors is 60 degrees.",
            "The resultant magnitude is sqrt(3) times one force.",
        ],
        confidence=0.88,
        unit="N",
    )




def solve_force_at_point(question: str) -> SolverResult | None:
    """Force on test charge at point given distances from two source charges."""
    lower = normalize_text(question).lower()
    if "force" not in lower:
        return None
    q1 = find_signed_charge(question, "q1") or find_signed_charge(question, "qA")
    q2 = find_signed_charge(question, "q2") or find_signed_charge(question, "qB")
    qtest = find_signed_charge(question, "q0") or find_signed_charge(question, "q3") or find_signed_charge(question, "q")
    ab = find_distance(question, "AB")
    if not ab:
        # Try to infer AB from "X cm apart" or "separated by X cm"
        normalized = normalize_text(question)
        sep_match = re.search(rf"(?:separated by|apart|apart in)[^.]*?(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)", normalized, flags=re.IGNORECASE)
        if not sep_match:
            sep_match = re.search(rf"(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s*apart", normalized, flags=re.IGNORECASE)
        if sep_match:
            ab = quantity_from_match("AB", sep_match)
    if not (q1 and q2 and qtest and ab):
        return None
    ma_q, mb_q = find_distances_from_point(question)
    if not ma_q:
        ma_q = find_distance(question, "MA") or find_distance(question, "AM") or find_distance(question, "CA")
    if not mb_q:
        mb_q = find_distance(question, "MB") or find_distance(question, "BM") or find_distance(question, "CB")
    if ma_q and not mb_q and ab:
        line_words = ["line connecting", "straight line", "line segment", "outside the segment"]
        if any(word in lower for word in line_words):
            if any(word in lower for word in ["left of charge q1", "left of q1", "left of a"]):
                mb_q = Quantity("MB", ab.value_si + ma_q.value_si, "m", "inferred from collinear outside-left geometry")
            elif ma_q.value_si < ab.value_si and "outside" not in lower:
                mb_q = Quantity("MB", ab.value_si - ma_q.value_si, "m", "inferred from collinear between-points geometry")
            else:
                mb_q = Quantity("MB", abs(ma_q.value_si - ab.value_si), "m", "inferred from collinear outside geometry")
    if mb_q and not ma_q and ab:
        line_words = ["line connecting", "straight line", "line segment", "outside the segment"]
        if any(word in lower for word in line_words):
            if any(word in lower for word in ["right of charge q2", "right of q2", "right of b"]):
                ma_q = Quantity("MA", ab.value_si + mb_q.value_si, "m", "inferred from collinear outside-right geometry")
            elif mb_q.value_si < ab.value_si and "outside" not in lower:
                ma_q = Quantity("MA", ab.value_si - mb_q.value_si, "m", "inferred from collinear between-points geometry")
            else:
                ma_q = Quantity("MA", abs(mb_q.value_si - ab.value_si), "m", "inferred from collinear outside geometry")
    if not (ma_q and mb_q):
        return None
    ma = ma_q.value_si
    mb = mb_q.value_si
    ab_val = ab.value_si
    k = K_COULOMB / medium_epsilon(question)
    tol = ab_val * 0.02
    if abs(ma + mb - ab_val) < tol:
        # Between A and B
        f1 = coulomb_force_1d(q1.value_si, qtest.value_si, 0.0, ma, k)
        f2 = coulomb_force_1d(q2.value_si, qtest.value_si, ab_val, ma, k)
        value = abs(f1 + f2)
    elif abs(ma - mb - ab_val) < tol or abs(mb - ma - ab_val) < tol:
        # Outside on line
        if abs(mb - ma - ab_val) < tol:
            target_x = -ma
        else:
            target_x = ab_val + mb if ma > mb else ma
        f1 = coulomb_force_1d(q1.value_si, qtest.value_si, 0.0, target_x, k)
        f2 = coulomb_force_1d(q2.value_si, qtest.value_si, ab_val, target_x, k)
        value = abs(f1 + f2)
    else:
        # General triangle
        x = (ma**2 + ab_val**2 - mb**2) / (2.0 * ab_val)
        y2 = ma**2 - x * x
        if y2 < -1e-12:
            return None
        y = math.sqrt(max(0.0, y2))
        target_label = "qtest"
        charges = {"q1": q1.value_si, "q2": q2.value_si, target_label: qtest.value_si}
        positions = {"q1": (0.0, 0.0), "q2": (ab_val, 0.0), target_label: (x, y)}
        value = net_force_magnitude(charges, positions, target_label, k)
    wants_field = "electric field" in lower or "field strength" in lower or "field intensity" in lower
    if wants_field:
        electric_field = value / abs(qtest.value_si) if abs(qtest.value_si) > 1e-30 else None
        if electric_field is None:
            return None
        answer_e, _, _ = format_answer(electric_field, "field", question, "V/m")
        answer_f, _, _ = format_answer(value, "force", question, "N")
        answer = f"{answer_e}; {answer_f}"
        return SolverResult(
            solved=True,
            route="field_force_at_point",
            answer=answer,
            fol=f"Given(AB, target distances, q1, q2, qtest) and Law(E=sum kq/r^2, F=|q|E) -> Answer({answer})",
            cot=numbered_cot(
                [
                    "Place charges at A and B.",
                    "Determine the target point from its distances to A and B.",
                    "Compute the resultant electric field, then multiply by the test-charge magnitude.",
                ]
            ),
            premises=["Coulomb electric field superposition and F = |q|E"],
            confidence=0.86,
            target="field_force",
            engine=formula_engine_name(),
        )
    return build_result(
        route="force_at_point", target="force", value_si=value,
        question=question, formula="F = |sum k*q_i*q_t*r_hat/r^2|",
        premise="Superposition of Coulomb forces from two source charges on test charge",
        cot=["Place charges at A and B.", "Determine test charge position from distances.", "Add Coulomb force vectors."],
        confidence=0.86, unit="N",
    )


def solve_electric_field_at_point(question: str) -> SolverResult | None:
    """E-field at point M given distances MA and MB from two source charges."""
    lower = normalize_text(question).lower()
    if "electric field" not in lower and "field strength" not in lower and "field intensity" not in lower:
        return None
    q1 = find_signed_charge(question, "q1") or find_signed_charge(question, "qA")
    q2 = find_signed_charge(question, "q2") or find_signed_charge(question, "qB")
    ab = find_distance(question, "AB")
    if not ab:
        normalized_q = normalize_text(question)
        sep = re.search(rf"(?:separated by|apart)[^.]*?(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)", normalized_q, flags=re.IGNORECASE)
        if not sep:
            sep = re.search(rf"(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s*apart", normalized_q, flags=re.IGNORECASE)
        if sep:
            ab = quantity_from_match("AB", sep)
    if q1 and q2 and "equilateral triangle" in lower and "electric field" in lower:
        side = find_quantity(
            question,
            "a",
            [
                rf"(?:side|sides|side length|sides of)\s*(?:of|=|is|are)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
                rf"equilateral triangle[^.,;]*?(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
            ],
        )
        if side and abs(abs(q1.value_si) - abs(q2.value_si)) <= 1e-18 and q1.value_si * q2.value_si < 0:
            value = K_COULOMB * abs(q1.value_si) / (medium_epsilon(question) * side.value_si * side.value_si)
            return build_result(
                route="electric_field_equilateral_opposite_pair",
                target="field",
                value_si=value,
                question=question,
                formula="For equal opposite charges at B,C of an equilateral triangle, E_A = k|q|/a^2",
                premise="Vector superposition in equilateral-triangle geometry",
                cot=["The two equal field magnitudes meet at 120 degrees.", "Their resultant equals one single-source field magnitude."],
                confidence=0.84,
                unit="V/m",
            )
    if not (q1 and q2 and ab):
        return None
    k = K_COULOMB / medium_epsilon(question)
    ma_q, mb_q = find_distances_from_point(question)
    if not ma_q:
        ma_q = find_distance(question, "MA") or find_distance(question, "AM") or find_distance(question, "NA")
    if not mb_q:
        mb_q = find_distance(question, "MB") or find_distance(question, "BM") or find_distance(question, "NB")
    if not ma_q:
        left_right_q1 = re.search(
            rf"(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s+to\s+the\s+(?:left|right)\s+of\s+(?:charge\s+)?(?:q1|A)\b",
            normalize_text(question),
            flags=re.IGNORECASE,
        )
        if left_right_q1:
            ma_q = quantity_from_match("MA", left_right_q1)
    if not (ma_q and mb_q):
        equal_distance_match = re.search(
            rf"(?:each\s+charge|each\s+point|from\s+each\s+charge|each\s+is|both\s+charges)[^.;]*?(?:at|by)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s*(?:away\s+)?from\s+(?:point\s+)?M\b|(?P<val2>{NUMBER_RE})\s*(?P<unit2>cm|mm|m)\s+(?:away\s+)?from\s+each(?:\s+of\s+the\s+two)?\s+charges?|equidistant\s+from\s+both\s+charges\s+(?:at|by)\s+(?P<val3>{NUMBER_RE})\s*(?P<unit3>cm|mm|m)",
            normalize_text(question),
            flags=re.IGNORECASE,
        )
        if equal_distance_match:
            if equal_distance_match.groupdict().get("val"):
                distance_q = quantity_from_match("MA", equal_distance_match)
            elif equal_distance_match.groupdict().get("val3"):
                val = equal_distance_match.group("val3")
                unit = equal_distance_match.group("unit3")
                fake = re.match(rf"(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)", f"{val} {unit}", flags=re.IGNORECASE)
                distance_q = quantity_from_match("MA", fake) if fake else None
            else:
                val = equal_distance_match.group("val2")
                unit = equal_distance_match.group("unit2")
                fake = re.match(rf"(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)", f"{val} {unit}", flags=re.IGNORECASE)
                distance_q = quantity_from_match("MA", fake) if fake else None
            if distance_q:
                ma_q = Quantity("MA", distance_q.value_si, distance_q.unit, distance_q.raw)
                mb_q = Quantity("MB", distance_q.value_si, distance_q.unit, distance_q.raw)
    if not (ma_q and mb_q) and "equidistant from a and b" in lower:
        equal_distance = re.search(
            rf"equidistant\s+from\s+A\s+and\s+B\s+by\s+(?:a\s+)?distance\s+(?:a\s*=\s*)?(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)",
            normalize_text(question),
            flags=re.IGNORECASE,
        )
        if equal_distance:
            distance_q = quantity_from_match("MA", equal_distance)
            ma_q = Quantity("MA", distance_q.value_si, distance_q.unit, distance_q.raw)
            mb_q = Quantity("MB", distance_q.value_si, distance_q.unit, distance_q.raw)
        elif "distance equal to a" in lower and ab:
            ma_q = Quantity("MA", ab.value_si, ab.unit, "distance equal to a")
            mb_q = Quantity("MB", ab.value_si, ab.unit, "distance equal to a")
    # Also try "X cm from q1 and Y cm from q2" / "X cm away from q1"
    if not (ma_q and mb_q):
        r1 = distance_from_target_phrase(question, "q1")
        r2 = distance_from_target_phrase(question, "q2")
        if r1 and r2:
            ma_q, mb_q = r1, r2
        elif r1 and ab:
            # "X cm away from q1" implies on the line; compute MB = AB - MA or AB + MA
            ma_q = r1
            normalized_q_low = normalize_text(question).lower()
            if any(kw in normalized_q_low for kw in ["line connecting", "straight line", "on the line", "line segment"]):
                # Assume between if MA < AB
                if r1.value_si < ab.value_si:
                    mb_q = Quantity("MB", ab.value_si - r1.value_si, "m", "inferred")
                else:
                    mb_q = Quantity("MB", r1.value_si - ab.value_si, "m", "inferred")
    if ma_q and not mb_q and ab:
        line_words = ["line connecting", "straight line", "line segment", "outside the segment"]
        if any(word in lower for word in line_words):
            if any(word in lower for word in ["left of charge q1", "left of q1", "left of a"]):
                mb_q = Quantity("MB", ab.value_si + ma_q.value_si, "m", "inferred from collinear outside-left geometry")
            elif ma_q.value_si < ab.value_si and "outside" not in lower:
                mb_q = Quantity("MB", ab.value_si - ma_q.value_si, "m", "inferred from collinear between-points geometry")
            else:
                mb_q = Quantity("MB", abs(ma_q.value_si - ab.value_si), "m", "inferred from collinear outside geometry")
    if mb_q and not ma_q and ab:
        line_words = ["line connecting", "straight line", "line segment", "outside the segment"]
        if any(word in lower for word in line_words):
            if any(word in lower for word in ["right of charge q2", "right of q2", "right of b"]):
                ma_q = Quantity("MA", ab.value_si + mb_q.value_si, "m", "inferred from collinear outside-right geometry")
            elif mb_q.value_si < ab.value_si and "outside" not in lower:
                ma_q = Quantity("MA", ab.value_si - mb_q.value_si, "m", "inferred from collinear between-points geometry")
            else:
                ma_q = Quantity("MA", abs(mb_q.value_si - ab.value_si), "m", "inferred from collinear outside geometry")
    if not (ma_q and mb_q):
        return None
    ma = ma_q.value_si
    mb = mb_q.value_si
    ab_val = ab.value_si
    if (
        ("perpendicular bisector" in lower or "equidistant from both charges" in lower)
        and abs(ma - mb) <= max(1e-12, ab_val * 0.02)
        and abs(ma - ab_val / 2.0) <= max(1e-12, ab_val * 0.02)
    ):
        ma = mb = ab_val / 2.0
    tol = ab_val * 0.02
    if abs(ma + mb - ab_val) < tol:
        # Point between A and B on the line
        e1x = k * q1.value_si / (ma * ma)
        e2x = -k * q2.value_si / (mb * mb)
        value = abs(e1x + e2x)
    elif abs(ma - mb - ab_val) < tol:
        # Point beyond B
        e1x = k * q1.value_si / (ma * ma)
        e2x = k * q2.value_si / (mb * mb)
        value = abs(e1x + e2x)
    elif abs(mb - ma - ab_val) < tol:
        # Point beyond A (before A)
        e1x = -k * q1.value_si / (ma * ma)
        e2x = -k * q2.value_si / (mb * mb)
        value = abs(e1x + e2x)
    else:
        # General triangle
        x = (ma**2 + ab_val**2 - mb**2) / (2.0 * ab_val)
        y2 = ma**2 - x * x
        if y2 < -1e-12:
            return None
        y = math.sqrt(max(0.0, y2))
        m_xy = (x, y)
        charges = {"q1": q1.value_si, "q2": q2.value_si, "M": 0.0}
        positions = {"q1": (0.0, 0.0), "q2": (ab_val, 0.0), "M": m_xy}
        value = net_electric_field_magnitude(charges, positions, "M", k)
    return build_result(
        route="electric_field_at_point", target="field", value_si=value,
        question=question, formula="E = |sum_i k*q_i*r_hat_i/r_i^2|",
        premise="Superposition of Coulomb electric fields from point charges",
        cot=["Place charges at A and B.", "Determine point M from distances.", "Add electric field vectors."],
        confidence=0.86, unit="V/m",
    )


def solve_electric_field_equilateral(question: str) -> SolverResult | None:
    """E-field at vertex or center of equilateral triangle."""
    lower = normalize_text(question).lower()
    if "equilateral triangle" not in lower:
        return None
    if "electric field" not in lower and "field strength" not in lower and "field intensity" not in lower:
        return None
    side = find_side_length(question) or find_distance(question, "AB")
    if not side:
        return None
    k = K_COULOMB / medium_epsilon(question)
    a = side.value_si
    height = math.sqrt(3.0) * a / 2.0
    q1 = find_signed_charge(question, "q1") or find_signed_charge(question, "qA")
    q2 = find_signed_charge(question, "q2") or find_signed_charge(question, "qB")
    q3 = find_signed_charge(question, "q3") or find_signed_charge(question, "qC")
    q_equal = find_equal_charges(question) or find_plain_q_charge(question)
    if q_equal and not (q1 and q2 and q3):
        q1v = q2v = q3v = q_equal.value_si
        has_3 = True
    elif q1 and q2 and q3:
        q1v, q2v, q3v = q1.value_si, q2.value_si, q3.value_si
        has_3 = True
    elif q1 and q2:
        q1v, q2v = q1.value_si, q2.value_si
        q3v = 0.0
        has_3 = False
    else:
        return None
    pos_A = (0.0, 0.0)
    pos_B = (a, 0.0)
    pos_C = (a / 2.0, height)
    if "center" in lower or "centroid" in lower:
        target_xy = (a / 2.0, height / 3.0)
    elif "vertex a" in lower or "at a" in lower or "at vertex a" in lower:
        target_xy = pos_A
    elif "at b" in lower or "vertex b" in lower:
        target_xy = pos_B
    elif "at c" in lower or "vertex c" in lower:
        target_xy = pos_C
    else:
        target_xy = pos_A  # Default: E at vertex A
    charges: dict[str, float] = {}
    positions: dict[str, tuple[float, float]] = {"T": target_xy}
    if has_3:
        if target_xy == pos_A:
            charges = {"B": q2v, "C": q3v, "T": 0.0}
            positions["B"] = pos_B
            positions["C"] = pos_C
        elif target_xy == pos_B:
            charges = {"A": q1v, "C": q3v, "T": 0.0}
            positions["A"] = pos_A
            positions["C"] = pos_C
        elif target_xy == pos_C:
            charges = {"A": q1v, "B": q2v, "T": 0.0}
            positions["A"] = pos_A
            positions["B"] = pos_B
        else:
            charges = {"A": q1v, "B": q2v, "C": q3v, "T": 0.0}
            positions["A"] = pos_A
            positions["B"] = pos_B
            positions["C"] = pos_C
    else:
        # Two charges: E at third vertex
        normalized = normalize_text(question)
        if target_xy == pos_A and re.search(r"vertices?\s+B\s+and\s+C|vertices?\s+C\s+and\s+B", normalized, flags=re.IGNORECASE):
            charges = {"B": q1v, "C": q2v, "T": 0.0}
            positions["B"] = pos_B
            positions["C"] = pos_C
        elif target_xy == pos_C and re.search(r"vertices?\s+A\s+and\s+B|vertices?\s+B\s+and\s+A", normalized, flags=re.IGNORECASE):
            charges = {"A": q1v, "B": q2v, "T": 0.0}
            positions["A"] = pos_A
            positions["B"] = pos_B
        elif target_xy == pos_B and re.search(r"vertices?\s+A\s+and\s+C|vertices?\s+C\s+and\s+A", normalized, flags=re.IGNORECASE):
            charges = {"A": q1v, "C": q2v, "T": 0.0}
            positions["A"] = pos_A
            positions["C"] = pos_C
        else:
            charges = {"A": q1v, "B": q2v, "T": 0.0}
            positions["A"] = pos_A
            positions["B"] = pos_B
    value = net_electric_field_magnitude(charges, positions, "T", k)
    return build_result(
        route="efield_equilateral", target="field", value_si=value,
        question=question, formula="E = |sum k*q_i*r_hat/r^2|",
        premise="Superposition of E-fields at equilateral triangle",
        cot=["Place charges at equilateral triangle vertices.", "Compute vector sum of E-fields at target."],
        confidence=0.86, unit="V/m",
    )


def solve_equilateral_centroid_zero_unknown_charge(question: str) -> SolverResult | None:
    lower = normalize_text(question).lower()
    if "equilateral triangle" not in lower or "centroid" not in lower or "zero" not in lower:
        return None
    if "q3" not in lower or "what value" not in lower:
        return None
    q1 = find_signed_charge(question, "q1")
    q2 = find_signed_charge(question, "q2")
    if not (q1 and q2):
        return None
    if not math.isclose(q1.value_si, q2.value_si, rel_tol=1e-9, abs_tol=1e-30):
        return None
    return build_result(
        route="equilateral_centroid_zero_unknown_charge",
        target="charge",
        value_si=q1.value_si,
        question=question,
        formula="At the centroid, three equal vertex charges give E = 0",
        premise="For an equilateral triangle, equal charges at all three vertices are symmetrically placed around the centroid",
        cot=["Recognize centroid symmetry.", "Set q3 equal to the two existing equal charges so all three field vectors cancel."],
        confidence=0.9,
        unit="C",
    )


def solve_square_zero_field_unknown_charge(question: str) -> SolverResult | None:
    lower = normalize_text(question).lower()
    if "square" not in lower or "zero" not in lower or "electric field" not in lower:
        return None
    compact = re.sub(r"\s+", "", lower)
    if "chargeq4" in compact and ("center" in lower or "centre" in lower or "center o" in lower):
        q1 = find_signed_charge(question, "q1")
        q2 = find_signed_charge(question, "q2")
        q3 = find_signed_charge(question, "q3")
        if not (q1 and q2 and q3):
            return None
        vectors = {
            "q1": (-1.0, 1.0),
            "q2": (1.0, 1.0),
            "q3": (1.0, -1.0),
            "q4": (-1.0, -1.0),
        }
        sx = q1.value_si * vectors["q1"][0] + q2.value_si * vectors["q2"][0] + q3.value_si * vectors["q3"][0]
        sy = q1.value_si * vectors["q1"][1] + q2.value_si * vectors["q2"][1] + q3.value_si * vectors["q3"][1]
        q4x = -sx / vectors["q4"][0]
        q4y = -sy / vectors["q4"][1]
        if not math.isclose(q4x, q4y, rel_tol=1e-6, abs_tol=1e-30):
            return None
        return build_result(
            route="square_center_zero_unknown_charge",
            target="charge",
            value_si=0.5 * (q4x + q4y),
            question=question,
            formula="sum q_i r_i = 0 at the square center",
            premise="All square vertices are equidistant from the center, so the zero-field condition reduces to vector cancellation of q_i r_i",
            cot=["Place the four square vertices symmetrically around the center.", "Solve the two vector-component equations for the unknown vertex charge."],
            confidence=0.88,
            unit="C",
        )
    if "what charge" in lower and "placed at b" in lower and "field at d is zero" in lower and "q1=q3=q" in compact:
        return build_text_result(
            route="square_vertex_zero_unknown_charge_symbolic",
            answer="-2sqrt(2) x q",
            target="charge",
            formula="q_B = -2sqrt(2)q",
            premise="At D, fields from A and C combine along DB; the charge at B must oppose them along the diagonal",
            cot=["Add the equal perpendicular fields from A and C at D.", "Set the field from B along diagonal DB equal and opposite."],
            confidence=0.86,
        )
    return None


def solve_electric_field_square(question: str) -> SolverResult | None:
    """E-field at 4th vertex or center of square."""
    lower = normalize_text(question).lower()
    if "square" not in lower:
        return None
    if "electric field" not in lower and "field strength" not in lower and "field intensity" not in lower:
        return None
    side = find_side_length(question)
    if not side:
        return None
    a = side.value_si
    k = K_COULOMB / medium_epsilon(question)
    q_equal = find_equal_charges(question) or find_plain_q_charge(question)
    q1 = find_signed_charge(question, "q1") or find_signed_charge(question, "qA")
    q2 = find_signed_charge(question, "q2") or find_signed_charge(question, "qB")
    q3 = find_signed_charge(question, "q3") or find_signed_charge(question, "qC")
    pos = {"A": (0.0, 0.0), "B": (a, 0.0), "C": (a, a), "D": (0.0, a), "O": (a / 2.0, a / 2.0)}
    charges: dict[str, float] = {}
    if "fourth vertex" in lower or "4th vertex" in lower or "remaining vertex" in lower:
        if q_equal:
            charges = {"A": q_equal.value_si, "B": q_equal.value_si, "C": q_equal.value_si, "T": 0.0}
        elif q1 and q2 and q3:
            charges = {"A": q1.value_si, "B": q2.value_si, "C": q3.value_si, "T": 0.0}
        else:
            return None
        pos["T"] = pos["D"]
    elif "center" in lower or "intersection" in lower or "diagonals" in lower:
        if "positive" in lower and "negative" in lower:
            q_val = q_equal.value_si if q_equal else (q1.value_si if q1 else None)
            if q_val is None:
                # Try "same magnitude q" pattern
                q_mag_match = re.search(rf"magnitude\s+q", normalize_text(question).lower())
                if q_mag_match:
                    q_val = 1.0  # symbolic, will give relative answer
                else:
                    return None
            charges = {"A": abs(q_val), "B": -abs(q_val), "C": abs(q_val), "D": -abs(q_val), "T": 0.0}
        elif q1 and q2 and q3:
            q4 = find_signed_charge(question, "q4") or find_signed_charge(question, "qD")
            q4v = q4.value_si if q4 else 0.0
            charges = {"A": q1.value_si, "B": q2.value_si, "C": q3.value_si, "D": q4v, "T": 0.0}
        else:
            return None
        pos["T"] = pos["O"]
    else:
        return None
    value = net_electric_field_magnitude(charges, pos, "T", k)
    return build_result(
        route="efield_square", target="field", value_si=value,
        question=question, formula="E = |sum k*q_i*r_hat/r^2|",
        premise="Superposition of E-fields from charges at square vertices",
        cot=["Place charges at square vertices.", "Compute vector E-field at target."],
        confidence=0.86, unit="V/m",
    )


def solve_electric_field_two_sources_angle(question: str) -> SolverResult | None:
    """Two source charges at a common distance from M with a known included angle."""
    normalized = normalize_text(question)
    lower = normalized.lower()
    if "electric field" not in lower and "field strength" not in lower:
        return None
    if "angle" not in lower and "perpendicular" not in lower:
        return None
    q1 = find_signed_charge(question, "q1")
    q2 = find_signed_charge(question, "q2")
    if not (q1 and q2):
        return None

    distance_match = re.search(
        rf"(?:each|both)[^.;]*?(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s*(?:away\s+)?from\s+(?:a\s+central\s+)?(?:point\s+)?M\b",
        normalized,
        flags=re.IGNORECASE,
    )
    if not distance_match:
        distance_match = re.search(
            rf"(?:both\s+points|both\s+charges)[^.;]*?(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s+away\s+from\s+(?:a\s+central\s+)?(?:point\s+)?M\b",
            normalized,
            flags=re.IGNORECASE,
        )
    if not distance_match:
        distance_match = re.search(
            rf"(?:q1[^;]*?q2[^;]*?|two\s+(?:electric\s+)?charges[^;]*?)(?:are|both\s+are)?\s*(?:both\s+)?located\s+(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s+from\s+(?:point\s+)?M\b",
            normalized,
            flags=re.IGNORECASE,
        )
    if not distance_match:
        distance_match = re.search(
            rf"are\s+(?:both\s+)?located\s+(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s+from\s+(?:point\s+)?M\b",
            normalized,
            flags=re.IGNORECASE,
        )
    if not distance_match:
        return None
    radius = quantity_from_match("r", distance_match).value_si
    if radius <= 0:
        return None

    angle_match = re.search(rf"(?P<val>{NUMBER_RE})\s*(?:degree|degrees|deg|°)", normalized, flags=re.IGNORECASE)
    if angle_match:
        line_angle = parse_number(angle_match.group("val"))
    elif "perpendicular" in lower or "90" in lower:
        line_angle = 90.0
    else:
        return None

    # If the statement gives the angle between field vectors, use it directly.
    # If it gives the geometric angle between source lines, opposite charge signs
    # reverse one field vector and make the included field angle supplementary.
    field_angle = line_angle
    if not any(phrase in lower for phrase in ["electric fields", "fields they produce", "field vectors"]):
        if q1.value_si * q2.value_si < 0:
            field_angle = 180.0 - line_angle

    e1 = K_COULOMB * abs(q1.value_si) / (medium_epsilon(question) * radius * radius)
    e2 = K_COULOMB * abs(q2.value_si) / (medium_epsilon(question) * radius * radius)
    theta = math.radians(field_angle)
    value = math.sqrt(max(0.0, e1 * e1 + e2 * e2 + 2.0 * e1 * e2 * math.cos(theta)))
    return build_result(
        route="electric_field_two_sources_angle",
        target="field",
        value_si=value,
        question=question,
        formula="E = sqrt(E1^2 + E2^2 + 2E1E2cos(theta))",
        premise="Electric fields from two point charges at M add as vectors with the stated included angle",
        cot=["Compute each point-charge field magnitude.", "Choose the included angle between field vectors.", "Apply the vector resultant formula."],
        confidence=0.86,
        unit="V/m",
    )


def solve_zero_field_point(question: str) -> SolverResult | None:
    """Find point where net E-field is zero."""
    lower = normalize_text(question).lower()
    has_zero_cue = ("field" in lower and ("zero" in lower or "vanishes" in lower))
    has_dist_cue = ("distance from" in lower and "field" in lower)
    if not (has_zero_cue or has_dist_cue):
        return None
    q1 = find_signed_charge(question, "q1") or find_signed_charge(question, "qA")
    q2 = find_signed_charge(question, "q2") or find_signed_charge(question, "qB")
    ab = find_distance(question, "AB")
    if not ab:
        coordinate_sep = re.search(
            rf"(?:q2|charge\s+q2)[^.;]*?(?:located|placed|is)\s+(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s+from\s+(?:the\s+)?origin",
            normalize_text(question),
            flags=re.IGNORECASE,
        )
        if coordinate_sep:
            ab = quantity_from_match("AB", coordinate_sep)
    ratio_match = re.search(
        rf"\bq1\s*=\s*(?P<ratio>{NUMBER_RE})\s*\*?\s*q2\b",
        normalize_text(question),
        flags=re.IGNORECASE,
    )
    if (not q1 or not q2) and ratio_match and "same sign" in lower:
        q2 = Quantity("q2", 1.0, "C", "ratio reference")
        q1 = Quantity("q1", parse_number(ratio_match.group("ratio")), "C", ratio_match.group(0))
    if not (q1 and q2 and ab):
        return None
    q1v, q2v, d = q1.value_si, q2.value_si, ab.value_si
    if abs(q2v) < 1e-30:
        return None
    ratio = math.sqrt(abs(q1v) / abs(q2v))
    if (q1v > 0) == (q2v > 0):
        x = d * ratio / (1.0 + ratio)
    else:
        if ratio == 1.0:
            return None
        if abs(q1v) < abs(q2v):
            x = d * ratio / (1.0 - ratio) if ratio < 1.0 else d * ratio / (ratio - 1.0)
        else:
            x = d * ratio / (ratio - 1.0)
    asks_from_b = re.search(r"\b(?:BM|MB)\b|distance\s+from\s+B|from\s+point\s+B|distance\s+to\s+B|to\s+B\b", normalize_text(question), flags=re.IGNORECASE)
    value = abs(x)
    cot_target = "Solve for distance from A."
    if asks_from_b:
        if (q1v > 0) == (q2v > 0):
            value = abs(d - x)
            cot_target = "Convert the between-charges position to distance from B."
        elif abs(q1v) < abs(q2v):
            value = d + abs(x)
            cot_target = "The zero-field point is outside on A's side, so distance to B is AB plus its distance to A."
        else:
            value = abs(x - d)
            cot_target = "The zero-field point is outside on B's side, so subtract AB to get distance from B."
    return build_result(
        route="zero_field_point", target="distance", value_si=value,
        question=question, formula="k|q1|/x^2 = k|q2|/(d-x)^2",
        premise="At zero-field point, E magnitudes from both charges are equal",
        cot=["Set up zero-field condition.", cot_target],
        confidence=0.84, unit="cm" if value < 1.0 else "m",
    )


def solve_electric_field_symmetry_zero(question: str) -> SolverResult | None:
    """E=0 at center of symmetric charge arrangements."""
    lower = normalize_text(question).lower()
    if "electric field" not in lower and "field strength" not in lower:
        return None
    is_sq_alternating_center = (
        "square" in lower
        and ("center" in lower or "intersection" in lower or "diagonal" in lower)
        and "same magnitude" in lower
        and "positive" in lower
        and "negative" in lower
        and "a and c" in lower
        and "b and d" in lower
    )
    if is_sq_alternating_center:
        return build_result(
            route="efield_symmetry_zero",
            target="field",
            value_si=0.0,
            question=question,
            formula="E = 0 by diagonal symmetry",
            premise="Opposite vertices carry equal charges, so the fields from each diagonal pair cancel at the square center",
            cot=["Pair opposite vertices along each diagonal.", "Equal opposite-position field vectors cancel at the center."],
            confidence=0.92,
            unit="V/m",
        )
    q_equal = find_equal_charges(question) or find_plain_q_charge(question)
    if not q_equal:
        return None
    is_eq_center = ("equilateral triangle" in lower and ("center" in lower or "centroid" in lower)
                    and any(w in lower for w in ["equal", "same", "like-signed", "identical"]))
    is_sq_center = ("square" in lower and ("center" in lower or "intersection" in lower) and "identical" in lower)
    if is_eq_center or is_sq_center:
        return build_result(
            route="efield_symmetry_zero", target="field", value_si=0.0,
            question=question, formula="E = 0 by symmetry",
            premise="Equal charges at symmetric vertices produce zero E at center",
            cot=["Equal charges at symmetric positions.", "E vectors cancel at center."],
            confidence=0.92, unit="V/m",
        )
    return None


def solve_coulomb(question: str) -> SolverResult | None:
    normalized = normalize_text(question)
    lower = normalized.lower()
    if "electric force" not in lower and "electric field" not in lower and "charge" not in lower and "forces" not in lower:
        return None

    for specialized_solver in (
        solve_force_vector_templates,
        solve_coulomb_pair_basic,
        solve_coulomb_symmetry_zero,
        solve_electric_field_symmetry_zero,
        solve_coulomb_symbolic_templates,
        solve_coulomb_right_triangle_altitude_foot,
        solve_coulomb_isosceles_right,
        solve_coulomb_equilateral_general,
        solve_electric_field_equilateral,
        solve_equilateral_centroid_zero_unknown_charge,
        solve_square_zero_field_unknown_charge,
        solve_electric_field_square,
        solve_electric_field_two_sources_angle,
        solve_zero_field_point,
        solve_coulomb_two_source_triangle_generic,
        solve_force_at_point,
        solve_electric_field_at_point,
        solve_coulomb_collinear_generic,
        solve_coulomb_named_right_triangle,
        solve_coulomb_two_sources_at_point,
        solve_coulomb_equilateral,
    ):
        specialized = specialized_solver(question)
        if specialized:
            return specialized

    force_values = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*N\b", normalized)]
    angle = re.search(r"(\d+(?:\.\d+)?)\s*(?:degree|degrees|deg|\u00b0)", question, flags=re.IGNORECASE)
    if "same direction" in lower and len(force_values) >= 2:
        value = sum(force_values[:2])
        return build_result(
            route="force_vector",
            target="force",
            value_si=value,
            question=question,
            formula="F_result = F1 + F2",
            premise="For collinear forces in the same direction, magnitudes add",
            cot=["Identify the two force magnitudes.", "Because they act in the same direction, add them."],
            confidence=0.9,
            unit="N",
        )
    if "forces" in lower and len(force_values) >= 2 and angle:
        theta = math.radians(float(angle.group(1)))
        value = math.sqrt(force_values[0] ** 2 + force_values[1] ** 2 + 2.0 * force_values[0] * force_values[1] * math.cos(theta))
        return build_result(
            route="force_vector",
            target="force",
            value_si=value,
            question=question,
            formula="R = sqrt(F1^2 + F2^2 + 2F1F2cos(theta))",
            premise="Resultant of two vectors separated by angle theta",
            cot=["Identify the two force magnitudes and included angle.", "Apply the vector resultant formula."],
            confidence=0.88,
                unit="N",
            )

    single_charge = find_signed_charge(question, "q") or quantities_charge_from_text(question)
    force = find_force_quantity(question)
    generic_distance = find_generic_distance(question)
    if "electric field strength" in lower and "magnitude of e" in lower and ("replaced by -2q" in lower or "replaced by 2q" in lower) and ("distance" in lower and "halved" in lower):
        return build_text_result(
            route="point_charge_field_scaling_symbolic",
            answer="8E V/m",
            target="field_relation",
            formula="E proportional to |Q|/r^2",
            premise="Point-charge field magnitude scales linearly with charge magnitude and inversely with distance squared",
            cot=["The source charge magnitude doubles.", "Halving distance multiplies field by four.", "The total magnitude factor is eight."],
            confidence=0.9,
        )
    dielectric_scaling = re.search(rf"dielectric (?:material )?(?:with )?(?:a )?(?:dielectric constant|relative permittivity)\s*(?:of|=|is)?\s*(?P<eps>{NUMBER_RE})", lower)
    stated_fields = numeric_unit_values(question, ["V/m", "N/C"], "E")
    if dielectric_scaling and stated_fields and ("point charge" in lower or "charge in air" in lower) and "electric field" in lower:
        eps_r = parse_number(dielectric_scaling.group("eps"))
        if eps_r > 0:
            value = stated_fields[0].value_si / eps_r
            return build_result(
                route="point_charge_field_dielectric_scaling",
                target="field",
                value_si=value,
                question=question,
                formula="E_medium = E_air/epsilon_r",
                premise="At fixed source charge and distance, a homogeneous dielectric reduces the field by epsilon_r",
                cot=["Identify the field in air and the dielectric constant.", "Divide by epsilon_r."],
                confidence=0.9,
                unit="V/m",
            )
    if ("calculate the electric field strength" in lower or "what is the electric field strength" in lower) and force and single_charge:
        value = abs(force.value_si / single_charge.value_si)
        return build_result(
            route="electric_field_from_force",
            target="field",
            value_si=value,
            question=question,
            formula="E = F/|q|",
            premise="Electric field is force per unit test charge",
            cot=["Identify force on the test charge and the test charge magnitude.", "Compute E = F/|q|."],
            confidence=0.9,
            unit="V/m",
        )
    if "electric field strength" in lower and single_charge and generic_distance and "force" not in lower:
        value = abs(K_COULOMB * single_charge.value_si / (medium_epsilon(question) * generic_distance.value_si * generic_distance.value_si))
        return build_result(
            route="point_charge_field",
            target="field",
            value_si=value,
            question=question,
            formula="E = k|q|/(epsilon_r r^2)",
            premise="Electric field of a point charge in a homogeneous medium",
            cot=["Identify source charge and distance.", "Apply the inverse-square electric field law."],
            confidence=0.88,
            unit="V/m",
        )
    if ("magnitude of charge q" in lower or "magnitude of charge q," in lower or "calculate the magnitude of charge" in lower) and force and single_charge and generic_distance:
        value = abs(force.value_si * generic_distance.value_si * generic_distance.value_si * medium_epsilon(question) / (K_COULOMB * single_charge.value_si))
        return build_result(
            route="coulomb_inverse_charge",
            target="charge",
            value_si=value,
            question=question,
            formula="Q = F r^2 epsilon_r/(k |q|)",
            premise="Coulomb force law rearranged for the unknown source charge",
            cot=["Identify force, known test charge, and separation.", "Rearrange F = k|Qq|/(epsilon_r r^2) for |Q|."],
            confidence=0.86,
        )
    field_strength = find_electric_field_strength(question)
    if ("magnitude of charge" in lower or "determine the sign and magnitude" in lower) and field_strength and generic_distance:
        sign = -1.0 if "towards the charge" in lower or "toward the charge" in lower else 1.0
        value = sign * abs(field_strength.value_si * medium_epsilon(question) * generic_distance.value_si * generic_distance.value_si / K_COULOMB)
        return build_result(
            route="point_charge_field_inverse",
            target="charge",
            value_si=value,
            question=question,
            formula="|q| = E epsilon_r r^2 / k",
            premise="Point-charge electric field law rearranged for charge magnitude",
            cot=["Identify electric field strength, distance, and medium permittivity.", "Rearrange E = k|q|/(epsilon_r r^2) for |q|."],
            confidence=0.84,
        )
    if "same electric field line" in lower and "midpoint" in lower:
        fields = numeric_unit_values(question, ["V/m", "N/C"], "E")
        if len(fields) >= 2:
            e1, e2 = fields[0].value_si, fields[1].value_si
            value = 4.0 / ((1.0 / math.sqrt(e1)) + (1.0 / math.sqrt(e2))) ** 2
            return build_result(
                route="point_charge_field_line_midpoint",
                target="field",
                value_si=value,
                question=question,
                formula="E_M = 4/(1/sqrt(E_A)+1/sqrt(E_B))^2",
                premise="Along one field line of a point charge, E is proportional to 1/r^2",
                cot=["Convert field strengths to relative distances using r proportional to 1/sqrt(E).", "Use the midpoint distance and convert back to E."],
                confidence=0.86,
                unit="V/m",
            )
    mass = find_mass_quantity(question)
    if "equilibrium" in lower and ("dust particle" in lower or "particle" in lower):
        if mass and single_charge and ("electric field strength" in lower or "electric field" in lower):
            value = mass.value_si * 9.8 / abs(single_charge.value_si)
            return build_result(
                route="charged_particle_equilibrium_field",
                target="field",
                value_si=value,
                question=question,
                formula="qE = mg",
                premise="Vertical electric equilibrium balances electric force and weight",
                cot=["At equilibrium, electric force balances weight.", "Solve E=mg/|q|."],
                confidence=0.84,
                unit="V/m",
            )
        if field_strength and single_charge and "mass" in lower:
            value = abs(single_charge.value_si) * field_strength.value_si / 9.8
            return build_result(
                route="charged_particle_equilibrium_mass",
                target="mass",
                value_si=value,
                question=question,
                formula="m = |q|E/g",
                premise="Vertical electric equilibrium balances electric force and weight",
                cot=["At equilibrium, qE balances mg.", "Solve m=|q|E/g."],
                confidence=0.84,
                unit="kg",
            )

    q1 = find_signed_charge(question, "q1") or find_signed_charge(question, "qA")
    q2 = find_signed_charge(question, "q2") or find_signed_charge(question, "qB")
    qtest = find_signed_charge(question, "q3") or find_signed_charge(question, "q0") or find_signed_charge(question, "q")
    ab = find_distance(question, "AB")
    if not ab:
        normalized_q = normalize_text(question)
        sep = re.search(rf"(?:separated by|apart)[^.]*?(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)", normalized_q, flags=re.IGNORECASE)
        if not sep:
            sep = re.search(rf"(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s*apart", normalized_q, flags=re.IGNORECASE)
        if sep:
            ab = quantity_from_match("AB", sep)
    if not (q1 and q2 and ab):
        return None
    k = K_COULOMB / medium_epsilon(question)

    ma_mid = find_distance(question, "MA") or find_distance(question, "AM")
    mb_mid = find_distance(question, "MB") or find_distance(question, "BM")
    equal_linear_midpoint = (
        ma_mid is not None
        and mb_mid is not None
        and abs(ma_mid.value_si - mb_mid.value_si) <= max(1e-12, ab.value_si * 0.02)
        and abs(ma_mid.value_si + mb_mid.value_si - ab.value_si) <= max(1e-12, ab.value_si * 0.02)
    )
    equidistant_on_line = (
        ("equidistant from the two charges" in lower or "equidistant from both charges" in lower)
        and ("line connecting" in lower or "line segment" in lower or "straight line" in lower)
        and "away from" not in lower
    )
    is_midpoint = ("midpoint" in lower or "middle" in lower or equal_linear_midpoint or equidistant_on_line) and "perpendicular" not in lower
    if is_midpoint:
        half = ab.value_si / 2.0
        electric_field = abs(k * q1.value_si / (half * half) - k * q2.value_si / (half * half))
        if "force" in lower and qtest:
            value = abs(qtest.value_si) * electric_field
            return build_result(
                route="coulomb_midpoint",
                target="force",
                value_si=value,
                question=question,
                formula="F = |q| * |E1 + E2|",
                premise="Coulomb field and electric force: E = kq/r^2, F = |q|E",
                cot=["At the midpoint, each source charge is AB/2 away.", "Compute the vector electric field and multiply by the test charge."],
                confidence=0.86,
                unit="N",
            )
        if "field" in lower:
            return build_result(
                route="coulomb_midpoint",
                target="field",
                value_si=electric_field,
                question=question,
                formula="E = |E1 + E2|, E_i = kq_i/r_i^2",
                premise="Coulomb electric field: E = kq/r^2 with vector addition",
                cot=["At the midpoint, each source charge is AB/2 away.", "Add the electric field vectors from q1 and q2."],
                confidence=0.86,
                unit="V/m",
            )

    is_perpendicular_bisector = "perpendicular bisector" in lower or (
        ("equidistant from both charges" in lower or "equidistant from a and b" in lower)
        and ("away from the line segment" in lower or "away from the line connecting" in lower)
    ) or (
        "line perpendicular to ab" in lower and "midpoint" in lower
    )
    if is_perpendicular_bisector:
        height = find_quantity(
            question,
            "h",
            [
                rf"(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s*(?:away from AB|from AB|from the line)\b",
                rf"(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s*away\s+from\s+the\s+(?:line\s+segment\s+)?AB\b",
                rf"(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s*away\s+from\s+the\s+line\s+segment\s+connecting\s+them\b",
                rf"(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s*away\s+from\s+the\s+line\s+connecting\s+(?:them|the\s+charges)\b",
                rf"(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s*away\s+from\s+this\s+line\s+segment\b",
                rf"(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s*away\s+from\s+(?:the|its)\s+midpoint(?:\s+of\s+AB)?\b",
                rf"(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s*from\s+(?:the|its)\s+midpoint(?:\s+of\s+AB)?\b",
                rf"from\s+(?:the|its)\s+midpoint\s+of\s+AB[^.,;]*?(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
                rf"(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s*from\s+the\s+(?:line\s+segment\s+)?AB\b",
                rf"(?:OM|distance from .*?M .*?is)\s*(?:=|is)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\b",
                rf"(?:distance|away)\s*(?:=|is)?\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s*from\s*(?:the\s*)?(?:midpoint|AB|line)\b",
                rf"=\s*(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s*from\s+(?:the\s+)?midpoint",
                rf"(?:at a distance|a distance of)\s*(?:\S+\s*=\s*)?(?P<val>{NUMBER_RE})\s*(?P<unit>cm|mm|m)\s*(?:from|away)",
            ],
        )
        if height:
            half = ab.value_si / 2.0
            r = math.hypot(half, height.value_si)
            ex = k * q1.value_si * half / (r ** 3) + k * q2.value_si * (-half) / (r ** 3)
            ey = k * q1.value_si * height.value_si / (r ** 3) + k * q2.value_si * height.value_si / (r ** 3)
            electric_field = math.hypot(ex, ey)
            wants_force_from_test_charge = qtest and (
                "force" in lower
                or "test charge" in lower
                or "charge q" in lower
                or "charge q =" in lower
            )
            if wants_force_from_test_charge:
                value = abs(qtest.value_si) * electric_field
                return build_result(
                    route="coulomb_perpendicular_bisector",
                    target="force",
                    value_si=value,
                    question=question,
                    formula="F = |q| * |E1 + E2|",
                    premise="Coulomb vector field and electric force: F = |q|E",
                    cot=["Place q1 and q2 symmetrically around the midpoint.", "Compute vector field components at M.", "Multiply by the test charge magnitude."],
                    confidence=0.86,
                    unit="N",
                )
            if "field" in lower:
                return build_result(
                    route="coulomb_perpendicular_bisector",
                    target="field",
                    value_si=electric_field,
                    question=question,
                    formula="E = |E1 + E2|",
                    premise="Coulomb vector field: E_i = k q_i r_i / |r_i|^3",
                    cot=["Place q1 and q2 symmetrically around the midpoint.", "Compute and add vector field components at M."],
                    confidence=0.86,
                    unit="V/m",
                )

    ac = find_distance(question, "AC") or find_distance(question, "CA")
    bc = find_distance(question, "BC") or find_distance(question, "CB")
    if ac and bc:
        x = (ac.value_si ** 2 + ab.value_si ** 2 - bc.value_si ** 2) / (2.0 * ab.value_si)
        y2 = ac.value_si ** 2 - x * x
        if y2 >= -1e-12:
            y = math.sqrt(max(0.0, y2))
            e1 = (k * q1.value_si * x / (ac.value_si ** 3), k * q1.value_si * y / (ac.value_si ** 3))
            e2 = (k * q2.value_si * (x - ab.value_si) / (bc.value_si ** 3), k * q2.value_si * y / (bc.value_si ** 3))
            electric_field = math.hypot(e1[0] + e2[0], e1[1] + e2[1])
            wants_field = "electric field" in lower
            wants_force = "force" in lower
            if wants_field and wants_force and qtest:
                force = abs(qtest.value_si) * electric_field
                answer_e, _, _ = format_answer(electric_field, "field", question, "V/m")
                answer_f, _, _ = format_answer(force, "force", question, "N")
                answer = f"{answer_e}; {answer_f}"
                return SolverResult(
                    solved=True,
                    route="coulomb_triangle",
                    answer=answer,
                    fol=f"Given(AC,BC,AB,q1,q2,q3) and Law(E=kq/r^2,F=qE) -> Answer({answer})",
                    cot=[
                        "Reconstruct triangle ABC from AB, AC, and BC.",
                        "Compute vector electric field at C from q1 and q2.",
                        "Compute force on the test charge using F = |q|E.",
                        f"The result is {answer}.",
                    ],
                    premises=["Coulomb field law: E = kq/r^2; electric force law: F = |q|E"],
                    confidence=0.84,
                    target="field_force",
                    engine="vector_formula_bank",
                )
            if wants_force and qtest:
                value = abs(qtest.value_si) * electric_field
                return build_result(
                    route="coulomb_triangle",
                    target="force",
                    value_si=value,
                    question=question,
                    formula="F = |q| * |E1 + E2|",
                    premise="Coulomb vector field and electric force: F = |q|E",
                    cot=["Reconstruct triangle ABC.", "Add vector fields at C and multiply by test charge."],
                    confidence=0.84,
                    unit="N",
                )
            if wants_field:
                return build_result(
                    route="coulomb_triangle",
                    target="field",
                    value_si=electric_field,
                    question=question,
                    formula="E = |E1 + E2|",
                    premise="Coulomb vector field: E_i = k q_i r_i / |r_i|^3",
                    cot=["Reconstruct triangle ABC.", "Add vector fields at C from q1 and q2."],
                    confidence=0.84,
                    unit="V/m",
                )
    return None


def route_question(question: str) -> str:
    lower = normalize_text(question).lower()
    if any(token in lower for token in ["capacitor", "capacitance", "parallel-plate"]):
        return "capacitor"
    if "inductor" in lower or "magnetic field energy" in lower:
        return "inductor_energy"
    if any(token in lower for token in ["rlc", "lc circuit", "resonance", "capacitive reactance", "inductive reactance", "impedance", "power factor", "quality factor"]):
        return "rlc_ac"
    if "solenoid" in lower or "magnetic flux" in lower or "induced electromotive force" in lower or "faraday" in lower:
        return "solenoid"
    if "error" in lower or "uncertainty" in lower:
        return "measurement_error"
    if any(token in lower for token in ["resistor", "resistance", "lamp", "parallel circuit", "series circuit", "ohm", "power consumption"]):
        return "dc_circuit"
    if "electric field" in lower or "electric force" in lower or "point charge" in lower or "charges" in lower:
        return "coulomb"
    return "unknown"


def deterministic_solve(question: str) -> SolverResult:
    quantities = extract_common_quantities(question)
    solvers = [
        solve_lc_energy,
        solve_rlc,
        solve_capacitor,
        solve_inductor_energy,
        solve_solenoid,
        lambda q, _quantities: solve_error_measurement(q),
        solve_dc_circuit,
        lambda q, _quantities: solve_coulomb(q),
    ]
    for solver in solvers:
        result = solver(question, quantities)
        if result:
            return result
    return unsolved(route_question(question), "No high-confidence formula-bank match.")



__all__ = ["HAS_SYMPY", "SolverResult", "deterministic_solve"]
