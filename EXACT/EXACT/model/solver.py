from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from .schemas import SolverResult
from .solvers.formula_bank import deterministic_solve
from .text_utils import compact_answer_text, numbered_cot


class DeterministicPhysicsSolver:
    """Native deterministic solver used by the modular pipeline."""

    def solve(self, question: str) -> SolverResult:
        try:
            result = deterministic_solve(question)
            data: dict[str, Any] = asdict(result) if is_dataclass(result) else dict(result)
        except Exception as exc:  # noqa: BLE001 - keep inference alive for audit.
            return SolverResult(
                solved=False,
                route="solver_error",
                answer="Unknown",
                fol="",
                cot=numbered_cot(["Parse the question.", f"Deterministic solver raised: {exc}"]),
                premises=[],
                confidence=0.2,
                target="unknown",
                warning=str(exc),
                engine="model_formula_bank",
            )

        return SolverResult(
            solved=bool(data.get("solved", False)),
            route=str(data.get("route", "unknown")),
            answer=compact_answer_text(data.get("answer", "Unknown")),
            fol=str(data.get("fol", "")),
            cot=numbered_cot(list(data.get("cot") or [])),
            premises=[str(item) for item in data.get("premises", [])],
            confidence=float(data.get("confidence", 0.0)),
            target=str(data.get("target", "unknown")),
            value_si=data.get("value_si"),
            unit=data.get("unit"),
            warning=data.get("warning"),
            engine=str(data.get("engine", "model_formula_bank")),
        )

