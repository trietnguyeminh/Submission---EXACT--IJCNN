from __future__ import annotations

import math
from typing import Any

from .schemas import SolverResult
from .text_utils import compact_answer_text, first_number, numbered_cot


def solver_response(result: SolverResult, question: str) -> dict[str, Any]:
    if result.solved:
        return {
            "answer": compact_answer_text(result.answer),
            "explanation": "The answer is computed by the deterministic physics solver from the quantities stated in the question.",
            "fol": result.fol,
            "cot": numbered_cot(result.cot),
            "premises": result.premises,
            "confidence": result.confidence,
        }
    return {
        "answer": "Unknown",
        "explanation": f"The current formula bank could not solve this question confidently: {result.warning}",
        "fol": "",
        "cot": numbered_cot(["Parse the question.", "No high-confidence matching formula was found."]),
        "premises": [],
        "confidence": 0.35,
    }


def normalize_physics_output(parsed: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    normalized = dict(parsed)
    answer = normalized.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        errors.append("answer must be a non-empty string")
        normalized["answer"] = "Unknown"
    else:
        normalized["answer"] = compact_answer_text(answer)

    explanation = normalized.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        errors.append("explanation must be a non-empty string")
        normalized["explanation"] = ""

    fol = normalized.get("fol", "")
    if isinstance(fol, list):
        fol = "; ".join(str(item) for item in fol)
    normalized["fol"] = str(fol)

    for key in ("cot", "premises"):
        value = normalized.get(key, [])
        if value is None:
            value = []
        if not isinstance(value, list):
            errors.append(f"{key} must be a list")
            value = [str(value)]
        normalized[key] = [str(item) for item in value]
    normalized["cot"] = numbered_cot(normalized.get("cot", []))

    confidence = normalized.get("confidence", None)
    if isinstance(confidence, (int, float)):
        normalized["confidence"] = max(0.0, min(1.0, float(confidence)))
    else:
        errors.append("confidence must be a number from 0.0 to 1.0")
        normalized["confidence"] = 0.0
    return normalized, errors


def arbitrate_response(
    response: dict[str, Any],
    solver_result: SolverResult,
    *,
    prefer_solver: bool,
) -> tuple[dict[str, Any], str]:
    if not solver_result.solved:
        return response, "llm_only_formula_missing"
    if not prefer_solver:
        return response, "llm_preferred"

    response_number = first_number(str(response.get("answer", "")))
    solver_number = first_number(solver_result.answer)
    if response_number is not None and solver_number is not None:
        if math.isclose(response_number, solver_number, rel_tol=0.04, abs_tol=1e-10):
            response["confidence"] = max(float(response.get("confidence", 0.0)), min(0.98, solver_result.confidence + 0.03))
            return response, "llm_agrees_with_solver"

    if solver_result.confidence >= 0.84:
        overridden = solver_response(solver_result, "")
        overridden["explanation"] = (
            str(response.get("explanation", "")).strip()
            or "The deterministic formula solver produced a high-confidence result."
        )
        overridden["confidence"] = max(float(overridden["confidence"]), solver_result.confidence)
        return overridden, "solver_override"
    return response, "llm_kept_solver_low_confidence"

