from __future__ import annotations

import re

from .schemas import QueryAnalysis
from .text_utils import normalize_text


class QueryAnalyzer:
    """Rule-based first pass used for metadata retrieval.

    This is intentionally simple and editable: it should route broad topic,
    target quantity, and geometry before the solver/LLM sees the question.
    """

    def analyze(self, question: str) -> QueryAnalysis:
        normalized = normalize_text(question)
        lower = normalized.lower()
        topic = self._topic(lower)
        target = self._target(lower)
        geometry_tags = self._geometry_tags(lower)
        signals = self._signals(lower)
        inverse = self._is_inverse(lower)
        return QueryAnalysis(
            question=question,
            normalized_question=normalized,
            topic=topic,
            target_quantity=target,
            geometry_tags=geometry_tags,
            signals=signals,
            inverse_or_constraint=inverse,
        )

    def _topic(self, lower: str) -> str:
        if self._has(lower, ["error", "uncertainty", "relative error", "absolute error", "measurement"]):
            return "measurement_error"
        if self._has(lower, ["parallel circuit", "series circuit", "lamp", "resistor", "resistance r", "power consumption"]):
            if not self._has(lower, ["rlc", "resonance", "reactance"]):
                return "dc_circuit"
        if self._has(lower, ["electric field", "field intensity", "field strength"]):
            return "electrostatics"
        if self._has(lower, ["charged ring", "circular ring", "charged rod", "non-conducting rod", "line charge density", "surface charge density", "charged disk", "semicircle"]):
            return "electrostatics"
        if self._has(lower, ["coulomb", "electric force", "electrostatic force", "point charges", "net force", "force acting on", "exert a force", "exerted by"]):
            return "electrostatics"
        if ("charge" in lower or "charges" in lower) and self._has(lower, ["triangle", "vertices", "midpoint", "perpendicular", "point a", "point b"]):
            return "electrostatics"
        if self._has(lower, ["capacitor", "capacitance", "dielectric", "plates", "charged under"]):
            return "capacitor"
        if self._has(lower, ["rlc", "resonance", "resonant", "reactance", "oscillation", "lc circuit", "angular frequency"]):
            return "ac_lc_rlc"
        if self._has(lower, ["inductor", "inductance", "magnetic energy", "energy stored in inductor"]):
            return "inductor"
        if self._has(lower, ["solenoid", "magnetic field", "magnetic flux", "faraday", "emf", "induced"]):
            return "magnetism_induction"
        return "unknown"

    def _target(self, lower: str) -> str:
        ordered = [
            ("electric_field", ["electric field", "field strength", "field intensity"]),
            ("force", ["force acting", "net force", "resultant force", "electric force", "electrostatic force"]),
            ("capacitance", ["capacitance", "capacitance c"]),
            ("charge", ["find q", "determine q", "value of q", "charge q", "what must q"]),
            ("voltage", ["voltage", "potential difference", "charged under u"]),
            ("energy", ["energy", "stored energy", "work done"]),
            ("frequency", ["frequency", "period", "angular frequency"]),
            ("reactance", ["reactance", "xc", "xl"]),
            ("magnetic_field", ["magnetic field", "magnetic induction"]),
            ("magnetic_flux", ["magnetic flux"]),
            ("emf", ["emf", "induced voltage"]),
            ("current", ["current"]),
            ("resistance", ["resistance", "impedance"]),
            ("distance", ["distance", "how far"]),
            ("error", ["error", "uncertainty"]),
        ]
        for target, phrases in ordered:
            if self._has(lower, phrases):
                return target
        if re.search(r"\bangle\b", lower):
            return "angle"
        return "unknown"

    def _geometry_tags(self, lower: str) -> list[str]:
        tags: list[str] = []
        if self._has(lower, ["equilateral triangle", "regular triangle"]):
            tags.append("equilateral_triangle")
        if self._has(lower, ["right triangle", "right-angled", "right angled"]):
            tags.append("right_triangle")
        if self._has(lower, ["straight line", "collinear", "same line", "on line"]):
            tags.append("collinear")
        if self._has(lower, ["midpoint", "middle point"]):
            tags.append("midpoint")
        if "perpendicular bisector" in lower:
            tags.append("perpendicular_bisector")
        if self._has(lower, ["square abcd", "square"]):
            tags.append("square")
        if self._has(lower, ["center", "centre", "centroid"]):
            tags.append("center")
        if self._has(lower, ["triangle", "vertices", "vertex"]):
            tags.append("triangle")
        if self._has(lower, ["ring", "rod", "disk", "sheet", "wire", "semicircle"]):
            tags.append("distributed_charge")
        if self._has(lower, ["uniform electric field", "homogeneous electric field"]):
            tags.append("uniform_field")
        if self._has(lower, ["rectangle abcd", "rectangle"]):
            tags.append("rectangle")
        return tags

    def _signals(self, lower: str) -> list[str]:
        signals: list[str] = []
        if re.search(r"\bq\s*['`]|qprime|q\s*0\b|qo\b", lower):
            signals.append("target_charge_notation")
        if self._has(lower, ["uC", "uc", "microc", "mf", "uf", "pf", "cm", "mm"]):
            signals.append("unit_conversion")
        if self._has(lower, ["direction", "vector"]):
            signals.append("vector_direction")
        if self._has(lower, ["zero", "equilibrium", "so that"]):
            signals.append("constraint")
        return signals

    def _is_inverse(self, lower: str) -> bool:
        return self._has(
            lower,
            [
                "find q",
                "determine q",
                "what must",
                "so that",
                "electric field is zero",
                "net electric field is zero",
                "given that the net",
                "find the angle",
            ],
        )

    @staticmethod
    def _has(lower: str, phrases: list[str]) -> bool:
        return any(phrase.lower() in lower for phrase in phrases)
