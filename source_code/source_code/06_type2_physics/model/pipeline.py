from __future__ import annotations

import math
import time
from pathlib import Path

from .analyzer import QueryAnalyzer
from .config import PipelineConfig
from .ft_corrector import LocalFtCorrector, build_corrector_payload, corrector_to_physics_response
from .io_utils import load_records, write_jsonl
from .kb import KnowledgeBase
from .llm_client import call_qwen, check_model, normalize_base_url
from .prompting import PromptBuilder
from .schemas import PipelineRecord, SolverResult
from .solver import DeterministicPhysicsSolver
from .solvers.formula_bank import find_distance, find_signed_charge, format_answer
from .text_utils import extract_json_object, first_number
from .validation import arbitrate_response, normalize_physics_output, solver_response


class PhysicsPipeline:
    def __init__(self, config: PipelineConfig, repo_root: Path | None = None):
        self.config = config
        self.repo_root = repo_root or Path(__file__).resolve().parents[1]
        self.analyzer = QueryAnalyzer()
        self.kb = KnowledgeBase(self.repo_root / config.kb_root)
        self.solver = DeterministicPhysicsSolver()
        self.prompt_builder = PromptBuilder()
        self.ft_corrector: LocalFtCorrector | None = None
        if config.generator == "ft-corrector":
            if config.adapter_path is None:
                raise ValueError("--adapter-path is required when --generator ft-corrector is used")
            self.ft_corrector = LocalFtCorrector(
                base_model=config.ft_base_model or config.model,
                adapter_path=config.adapter_path,
                max_new_tokens=config.ft_max_new_tokens,
            )

    def build_prompt_for_record(self, record: dict[str, object]) -> str:
        question = str(record["question"])
        analysis = self.analyzer.analyze(question)
        context = self.kb.retrieve(
            analysis,
            formula_top_k=self.config.formula_top_k,
            geometry_top_k=self.config.geometry_top_k,
            example_top_k=self.config.example_top_k,
        )
        solver_result = self.solver.solve(question)
        return self.prompt_builder.build(question, analysis, context, solver_result)

    @staticmethod
    def _same_quantity(left, right, *, rel_tol: float = 1e-6, abs_tol: float = 1e-15) -> bool:
        return bool(
            left
            and right
            and math.isclose(left.value_si, right.value_si, rel_tol=rel_tol, abs_tol=abs_tol)
        )

    def _solve_contextual_followup(
        self,
        question: str,
        previous_output: PipelineRecord | None,
    ) -> SolverResult | None:
        """Resolve narrow field-to-force follow-up rows that omit repeated geometry."""
        if previous_output is None or not previous_output.solver_result:
            return None
        lower = question.lower()
        if "force" not in lower or "placed at c" not in lower:
            return None
        if any(find_distance(question, label) for label in ("AC", "CA", "BC", "CB")):
            return None

        previous_solver = previous_output.solver_result
        if not previous_solver.get("solved") or "field" not in str(previous_solver.get("target", "")):
            return None

        q1 = find_signed_charge(question, "q1")
        q2 = find_signed_charge(question, "q2")
        qtest = find_signed_charge(question, "q3") or find_signed_charge(question, "q0")
        ab = find_distance(question, "AB")
        prev_q = previous_output.question
        if not (
            self._same_quantity(q1, find_signed_charge(prev_q, "q1"))
            and self._same_quantity(q2, find_signed_charge(prev_q, "q2"))
            and self._same_quantity(ab, find_distance(prev_q, "AB"), rel_tol=1e-6, abs_tol=1e-12)
            and qtest
        ):
            return None

        field_value = previous_solver.get("value_si")
        if field_value is None:
            field_value = first_number(str(previous_solver.get("answer", "")))
        if field_value is None:
            return None

        force_value = abs(qtest.value_si) * abs(float(field_value))
        answer, unit, _ = format_answer(force_value, "force", question, "N")
        return SolverResult(
            solved=True,
            route="contextual_field_force_followup",
            answer=answer,
            fol=f"Given(previous E_C, q3) and Law(F=|q3|E_C) -> Answer({answer})",
            cot=[
                "Reuse the electric field at point C from the immediately preceding matching source-charge problem.",
                "The current row supplies the test charge q3 but omits the repeated C geometry.",
                "Compute the force magnitude with F = |q3|E_C.",
            ],
            premises=[
                "The previous row has the same q1, q2, and AB geometry and asks for the electric field at C",
                "Electric force on a test charge is F = |q|E",
            ],
            confidence=0.86,
            target="force",
            value_si=force_value,
            unit=unit,
            warning="Used previous-row field context because this row omits the repeated point-C geometry.",
            engine="contextual_formula_bank",
        )

    def run_one(
        self,
        record: dict[str, object],
        item_index: int,
        base_url: str,
        previous_output: PipelineRecord | None = None,
    ) -> PipelineRecord:
        question = str(record["question"])
        started = time.time()
        analysis = self.analyzer.analyze(question)
        context = self.kb.retrieve(
            analysis,
            formula_top_k=self.config.formula_top_k,
            geometry_top_k=self.config.geometry_top_k,
            example_top_k=self.config.example_top_k,
        )
        solver_result = self.solver.solve(question)
        if not solver_result.solved:
            solver_result = self._solve_contextual_followup(question, previous_output) or solver_result
        prompt = self.prompt_builder.build(question, analysis, context, solver_result)
        raw_response: str | None = None
        validation_errors: list[str] = []

        if self.config.generator == "solver":
            response = solver_response(solver_result, question)
            valid_json = True
            arbitration = "solver_only"
        elif self.config.generator == "ft-corrector":
            baseline_response = solver_response(solver_result, question)
            payload = build_corrector_payload(
                question=question,
                analysis=analysis,
                retrieval=context,
                solver_result=solver_result,
                baseline_response=baseline_response,
            )
            assert self.ft_corrector is not None
            try:
                corrector_output, raw_response, validation_errors = self.ft_corrector.generate_json(payload)
            except Exception as exc:  # noqa: BLE001 - preserve a valid fallback output.
                corrector_output = {"accept_solver": True}
                raw_response = None
                validation_errors = [f"ft_corrector_failed: {exc}"]

            if validation_errors:
                response = baseline_response
                arbitration = "ft_invalid_fallback"
            elif corrector_output.get("accept_solver", False):
                response = baseline_response
                arbitration = "ft_accept_solver"
            else:
                response = corrector_to_physics_response(corrector_output, baseline_response)
                arbitration = "ft_corrected"
            valid_json = not validation_errors
        else:
            raw_response = call_qwen(
                base_url=base_url,
                model=self.config.model,
                prompt=prompt,
                timeout=self.config.timeout,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                disable_thinking=not self.config.enable_thinking,
                use_response_format=self.config.use_response_format,
            )
            try:
                parsed = extract_json_object(raw_response)
                response, validation_errors = normalize_physics_output(parsed)
            except Exception as exc:  # noqa: BLE001 - keep raw response for audit.
                response = solver_response(solver_result, question)
                validation_errors = [f"model_json_parse_failed: {exc}"]
            response, arbitration = arbitrate_response(
                response,
                solver_result,
                prefer_solver=self.config.prefer_solver,
            )
            valid_json = not validation_errors

        return PipelineRecord(
            item_index=item_index,
            sample_id=str(record.get("id")) if record.get("id") is not None else None,
            question=question,
            response=response,
            route=solver_result.route,
            solver_result=solver_result.to_dict(),
            arbitration=arbitration,
            valid_json=valid_json,
            validation_errors=validation_errors,
            raw_response=raw_response,
            latency_sec=round(time.time() - started, 3),
            analysis=analysis.to_dict(),
            retrieval=context.to_dict(),
        )

    def run_batch(self) -> int:
        records = load_records(
            self.config.input_path,
            self.config.question_column,
            self.config.id_column,
        )
        selected = records[self.config.start :]
        if self.config.limit is not None:
            selected = selected[: self.config.limit]
        if not selected:
            raise ValueError("No records selected")

        if self.config.dry_run:
            print(self.build_prompt_for_record(selected[0]))
            return 0

        base_url = normalize_base_url(self.config.base_url)
        if self.config.generator == "qwen" and not self.config.skip_model_check:
            check_model(base_url, self.config.model, self.config.timeout)

        if self.config.output_path.exists() and not self.config.append:
            self.config.output_path.unlink()
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)

        counts: dict[str, int] = {}
        previous_output: PipelineRecord | None = None
        for offset, record in enumerate(selected, start=self.config.start):
            output = self.run_one(record, offset, base_url, previous_output)
            write_jsonl(self.config.output_path, output.to_dict())
            previous_output = output
            counts[output.arbitration] = counts.get(output.arbitration, 0) + 1
            print(
                f"[{offset + 1}/{self.config.start + len(selected)}] "
                f"id={output.sample_id} route={output.route} arbitration={output.arbitration} "
                f"answer={output.response.get('answer')}"
            )
            if self.config.sleep > 0:
                time.sleep(self.config.sleep)

        print(f"Wrote {len(selected)} records to {self.config.output_path}")
        print(f"Arbitration counts: {counts}")
        return 0
