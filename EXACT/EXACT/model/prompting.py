from __future__ import annotations

import json

from .schemas import QueryAnalysis, RetrievedContext, SolverResult


PHYSICS_SYSTEM_PROMPT = """You are an EXACT 2026 Type 2 physics QA engine.
Solve from the input question only. Return exactly one valid JSON object and no markdown.

Required keys:
- answer: final numeric result plus unit only, e.g. "3.12 N" or "100 uF"
- explanation: concise natural-language explanation

Recommended keys:
- fol: one concise physics-logic formula string
- cot: list of concise public reasoning steps, each starting with "Step 1:", "Step 2:", ...
- premises: list of laws, formulas, or givens actually used
- confidence: number from 0.0 to 1.0

Rules:
- The answer field must not contain a sentence, direction, formula, or explanation. Put those in explanation/cot.
- If the final answer has a direction, keep answer as magnitude plus unit only and describe direction in explanation.
- Every cot item must start with the correct "Step n:" prefix.
- Prefer the deterministic solver result when it is provided and marked high confidence.
- Use retrieved formulas/geometry only when they match the question.
- Do not invent missing quantities. If the question is under-specified, lower confidence.
- Keep arithmetic consistent with SI conversions and the requested unit.
- IMPORTANT: You MUST attempt to solve the problem numerically. Do NOT return "Unknown" unless the question is truly impossible to answer. Use the retrieved formulas and your physics knowledge to compute the answer.
"""

PHYSICS_SOLVER_UNSOLVED_GUIDE = """
The deterministic solver could not solve this question, so YOU must solve it.
Follow these steps:
1. Identify all given quantities and convert to SI units (C, m, etc.)
2. Identify the target quantity (force, electric field, distance, etc.)
3. Select the appropriate physics formula:
   - Coulomb force: F = k|q1*q2|/r^2 (k = 9 x 10^9 N*m^2/C^2)
   - Electric field: E = k*q/r^2 (field from point charge)
   - Superposition: add E vectors from each charge
   - For geometry: use coordinate system, compute each vector component (Ex, Ey), then magnitude = sqrt(Ex^2 + Ey^2)
4. Compute the answer step by step
5. Express the result with the correct unit

Common geometry setups:
- Collinear: place charges on x-axis, E fields are 1D signed vectors
- Triangle: use law of cosines to find coordinates, add 2D field vectors
- Square: vertices at (0,0), (a,0), (a,a), (0,a); diagonal = a*sqrt(2)
- Equilateral triangle: vertices at (0,0), (a,0), (a/2, a*sqrt(3)/2)
- Perpendicular bisector: target at (0, h), sources at (-d/2, 0) and (d/2, 0)

You MUST provide a numeric answer. Do NOT return "Unknown".
"""


class PromptBuilder:
    def build(self, question: str, analysis: QueryAnalysis, context: RetrievedContext, solver: SolverResult) -> str:
        lines = [
            "/no_think",
            "Solve this EXACT Type 2 physics question.",
            "",
            "Question:",
            question,
            "",
            "Query analysis:",
            json.dumps(analysis.to_dict(), ensure_ascii=False),
            "",
            "Retrieved formula cards:",
        ]
        for card in context.formula_cards:
            payload = card.brief()
            lines.append(json.dumps(payload, ensure_ascii=False))
        lines.extend(["", "Retrieved geometry cards:"])
        for card in context.geometry_cards:
            payload = card.brief()
            lines.append(json.dumps(payload, ensure_ascii=False))
        if context.example_cards and (not solver.solved or solver.confidence < 0.84):
            lines.extend(["", "Retrieved abstract example templates:"])
            for card in context.example_cards:
                payload = card.brief()
                lines.append(json.dumps(payload, ensure_ascii=False))
        lines.extend(
            [
                "",
                "Deterministic solver context:",
                self._format_solver(solver),
            ]
        )
        # Add solving guidance when solver is unsolved
        if not solver.solved:
            lines.extend(["", PHYSICS_SOLVER_UNSOLVED_GUIDE.strip()])
        lines.extend(
            [
                "",
                "Return only this JSON shape:",
                "{",
                '  "answer": "numeric result plus unit only",',
                '  "explanation": "concise explanation; include direction here if needed",',
                '  "fol": "concise formula implication",',
                '  "cot": ["Step 1: ...", "Step 2: ..."],',
                '  "premises": ["formula or given used"],',
                '  "confidence": 0.0',
                "}",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _format_solver(solver: SolverResult) -> str:
        if not solver.solved:
            return (
                f"Solver status: unsolved. Engine: {solver.engine}. "
                f"Route: {solver.route}. Warning: {solver.warning}"
            )
        return "\n".join(
            [
                "Solver status: solved.",
                f"Engine: {solver.engine}",
                f"Route: {solver.route}",
                f"Answer: {solver.answer}",
                f"FOL/formula: {solver.fol}",
                f"Premises: {json.dumps(solver.premises, ensure_ascii=False)}",
                f"COT: {json.dumps(solver.cot, ensure_ascii=False)}",
                f"Confidence: {solver.confidence}",
            ]
        )
