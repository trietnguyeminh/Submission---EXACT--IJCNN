from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .analyzer import QueryAnalyzer
from .schemas import QueryAnalysis
from .text_utils import normalize_text


@dataclass
class PhysicsIR:
    """Lightweight audit IR for routing physics questions by law family.

    This is intentionally descriptive rather than executable.  It lets the
    knowledge audit report coverage by reusable physics patterns instead of by
    individual dataset row.
    """

    domain: str
    law_family: str
    target: str
    knowns: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    state_condition: list[str] = field(default_factory=list)
    geometry: list[str] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_physics_ir(question: str, analysis: QueryAnalysis | None = None) -> PhysicsIR:
    analyzer = QueryAnalyzer()
    analysis = analysis or analyzer.analyze(question)
    normalized = normalize_text(question)
    lower = normalized.lower()

    domain = _domain(lower, analysis.topic)
    law_family = _law_family(lower, domain)
    knowns = _known_symbols(normalized)
    target = _target(lower, analysis.target_quantity)
    state_condition = _state_conditions(lower)
    geometry = list(dict.fromkeys(analysis.geometry_tags + _geometry(lower)))
    signals = list(dict.fromkeys(analysis.signals + _signals(lower)))
    unknowns = [target] if target != "unknown" else []

    return PhysicsIR(
        domain=domain,
        law_family=law_family,
        target=target,
        knowns=knowns,
        unknowns=unknowns,
        state_condition=state_condition,
        geometry=geometry,
        signals=signals,
    )


def _domain(lower: str, topic: str) -> str:
    if _has(lower, ["capacitor", "capacitance", "parallel-plate", "dielectric", "plates"]):
        return "capacitor"
    if _has(lower, ["rlc", "lc circuit", "oscillat", "resonance", "reactance", "impedance", "power factor"]):
        return "ac_lc_rlc"
    if _has(lower, ["solenoid", "magnetic flux", "faraday", "induced", "magnetic field inside"]):
        return "magnetism_induction"
    if _has(lower, ["inductor", "inductance", "magnetic field energy"]):
        return "inductor"
    if _has(lower, ["error", "uncertainty", "least count", "measured", "actual", "true value"]):
        return "measurement_error"
    if _has(lower, ["resistor", "resistance", "lamp", "parallel circuit", "series circuit", "ohm", "power consumption"]):
        return "dc_circuit"
    if _has(lower, ["charged ring", "thin circular ring", "non-conducting rod", "straight wire", "semicircle", "circular disk", "sheet", "surface charge density"]):
        return "electrostatics"
    return topic


def _law_family(lower: str, domain: str) -> str:
    if domain == "capacitor":
        if _has(lower, ["breakdown", "maximum electric field", "dielectric strength", "emax"]):
            return "capacitor_breakdown"
        if _has(lower, ["disconnected", "remains connected", "connected to the source", "immersed", "moved apart"]):
            return "capacitor_state_change"
        if _has(lower, ["two capacitors", "c1", "c2", "series", "parallel", "like-polarity", "like-poled"]):
            return "capacitor_network"
        if _has(lower, ["parallel-plate", "parallel plate", "plate area", "circular plates", "radius"]):
            return "parallel_plate"
        return "capacitor_basic"
    if domain == "electrostatics":
        if _has(lower, ["ring", "rod", "disk", "sheet", "wire", "semicircle", "surface charge density", "line charge density"]):
            return "distributed_charge"
        if _has(lower, ["electron moves", "dust particle", "suspended", "thread", "string", "uniform electric field", "homogeneous electric field"]):
            return "uniform_field_motion"
        if _has(lower, ["relationship between e1 and e2", "relationship between the electric field strengths"]) or (
            "f1" in lower and "f2" in lower and "e1" in lower and "e2" in lower
        ):
            return "field_force_relation"
        if _has(lower, ["zero", "equilibrium", "what charge", "find q", "so that"]):
            return "zero_field_inverse"
        if _has(lower, ["triangle", "square", "midpoint", "perpendicular", "collinear", "vertices", "center"]):
            return "point_charge_geometry"
        return "point_charge_superposition"
    if domain == "ac_lc_rlc":
        if _has(lower, ["uam", "umb", "quadrature", "90 degrees out of phase", "lcω", "lcw", "lc omega"]):
            return "segmented_phasor"
        if _has(lower, ["reactance", "xl", "xc"]):
            return "reactance"
        if _has(lower, ["resonance", "resonant", "power factor", "quality factor"]):
            return "rlc_resonance"
        if _has(lower, ["energy", "maximum current", "instantaneous current"]):
            return "lc_energy_exchange"
        return "ac_impedance"
    if domain in {"inductor", "magnetism_induction"}:
        if _has(lower, ["solenoid"]):
            return "solenoid"
        if _has(lower, ["flux", "faraday", "induced"]):
            return "faraday_flux"
        return "inductor_energy"
    if domain == "measurement_error":
        if _has(lower, ["least count"]):
            return "least_count_error"
        if _has(lower, ["three", "measurements", "readings"]):
            return "repeated_measurement"
        if _has(lower, ["±", "+/-", "uncertainty"]):
            return "propagated_or_direct_uncertainty"
        return "true_vs_measured_error"
    if domain == "dc_circuit":
        if _has(lower, ["parallel"]):
            return "parallel_dc"
        if _has(lower, ["series"]):
            return "series_dc"
        return "ohm_power"
    return "unknown"


def _target(lower: str, target: str) -> str:
    if _has(lower, ["electric field", "field strength"]) and _has(lower, ["electric force", "force acting"]):
        return "field_force"
    if _has(lower, ["relationship between e1 and e2", "relationship between the electric field strengths"]) or (
        "f1" in lower and "f2" in lower and "e1" in lower and "e2" in lower
    ):
        return "field_relation"
    if _has(lower, ["impedance", " z "]):
        return "impedance"
    if _has(lower, ["power factor", "cosphi", "cos phi", "cosφ"]):
        return "power_factor"
    if _has(lower, ["quality factor"]):
        return "quality_factor"
    if _has(lower, ["dielectric constant", "relative permittivity"]):
        return "dielectric_constant"
    if _has(lower, ["maximum charge", "charge on the capacitor", "charge stored"]):
        return "charge"
    if _has(lower, ["average", "mean"]):
        return "mean_error"
    return target


def _known_symbols(text: str) -> list[str]:
    knowns: list[str] = []
    normalized = normalize_text(text)
    symbol_patterns = {
        "C": r"\bC\d*|capacitance",
        "U": r"\bU(?:AB|\d*)?|voltage|potential difference|RMS voltage|effective voltage",
        "Q": r"\bQ\d*|charge",
        "L": r"\bL\b|inductance",
        "I": r"\bI\d*|current",
        "R": r"\bR\d*|resistance",
        "Z": r"\bZ\b|impedance",
        "XL": r"\bX_?L\b|inductive reactance",
        "XC": r"\bX_?C\b|capacitive reactance",
        "f": r"\bf\b|frequency",
        "A": r"\bA\b|area|radius|circular plates",
        "d": r"\bd\b|distance|separation",
        "epsilon_r": r"epsilon|dielectric constant|relative permittivity",
        "Emax": r"emax|maximum electric field|dielectric strength",
        "B": r"\bB\b|magnetic field|flux density",
        "Phi": r"flux|phi",
    }
    for symbol, pattern in symbol_patterns.items():
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            knowns.append(symbol)
    return knowns


def _state_conditions(lower: str) -> list[str]:
    conditions: list[str] = []
    checks = {
        "disconnected": ["disconnected", "isolated"],
        "connected_source": ["connected to the source", "remains connected", "still connected"],
        "dielectric_inserted": ["dielectric", "immersed", "relative permittivity"],
        "plate_distance_changed": ["moved apart", "distance between", "doubled", "halved"],
        "resonance": ["resonance", "resonant", "at resonance"],
        "rms": ["rms", "effective"],
    }
    for label, phrases in checks.items():
        if _has(lower, phrases):
            conditions.append(label)
    return conditions


def _geometry(lower: str) -> list[str]:
    tags: list[str] = []
    checks = {
        "circular_plate": ["circular plates", "radius"],
        "parallel_plate": ["parallel-plate", "parallel plate"],
        "line_field": ["field line"],
        "right_isosceles": ["right isosceles"],
        "perpendicular_bisector": ["perpendicular bisector", "away from the line segment ab", "away from the line segment connecting"],
        "equidistant_ab": ["equidistant from a and b", "equidistant from both charges", "equidistant from the two charges"],
        "square_alternating_center": ["positive charges are located at a and c", "negative charges are located at b and d"],
        "distributed_axis": ["z-axis", "axis perpendicular", "center of the disk", "center of the ring"],
        "rectangle": ["rectangle abcd"],
        "uniform_field": ["uniform electric field", "homogeneous electric field"],
    }
    for label, phrases in checks.items():
        if _has(lower, phrases):
            tags.append(label)
    return tags


def _signals(lower: str) -> list[str]:
    signals: list[str] = []
    if re.search(r"\b(?:C|U|Q|R|I)\d+\b", lower):
        signals.append("indexed_variables")
    if _has(lower, ["sqrt", "pi", "cos"]):
        signals.append("symbolic_expression")
    if _has(lower, ["relationship", "compare", "directly proportional", "what happens"]):
        signals.append("conceptual")
    return signals


def _has(lower: str, phrases: list[str]) -> bool:
    return any(phrase.lower() in lower for phrase in phrases)
