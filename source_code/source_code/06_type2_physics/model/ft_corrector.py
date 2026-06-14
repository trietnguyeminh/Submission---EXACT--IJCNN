from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import QueryAnalysis, RetrievedContext, SolverResult
from .text_utils import compact_answer_text, extract_json_object, numbered_cot


CORRECTOR_SYSTEM_PROMPT = (
    "You are a physics answer corrector for EXACT Type 2. "
    "Decide whether to accept the deterministic solver/baseline answer or correct it. "
    "Return exactly one JSON object and no markdown."
)


def build_corrector_payload(
    *,
    question: str,
    analysis: QueryAnalysis | dict[str, Any],
    retrieval: RetrievedContext | dict[str, Any],
    solver_result: SolverResult | dict[str, Any],
    baseline_response: dict[str, Any],
) -> dict[str, Any]:
    if hasattr(analysis, "to_dict"):
        analysis_payload = analysis.to_dict()  # type: ignore[union-attr]
    else:
        analysis_payload = analysis
    if hasattr(retrieval, "to_dict"):
        retrieval_payload = retrieval.to_dict()  # type: ignore[union-attr]
    else:
        retrieval_payload = retrieval
    if hasattr(solver_result, "to_dict"):
        solver_payload = solver_result.to_dict()  # type: ignore[union-attr]
    else:
        solver_payload = solver_result
    return {
        "question": question,
        "analysis": analysis_payload,
        "retrieval": retrieval_payload,
        "solver_result": solver_payload,
        "baseline_response": baseline_response,
        "instructions": {
            "output_schema": {
                "accept_solver": "boolean",
                "error_type": "none | wrong_law | wrong_target | wrong_unit | wrong_cardinality | conceptual_text | arithmetic | missing_formula",
                "domain": "physics domain",
                "law_family": "law family",
                "target": "target quantity",
                "answer_cardinality": "integer number of final answer values",
                "corrected_answer": "final answer value(s) plus unit only",
                "premises": ["laws or givens used"],
                "confidence": "number from 0.0 to 1.0",
            },
            "rules": [
                "If the baseline is physically correct, set accept_solver=true.",
                "If correcting, set accept_solver=false and put the corrected value in corrected_answer.",
                "Do not include sample ids or dataset-specific reasoning.",
                "For multi-target questions, corrected_answer must contain all requested values separated by semicolons.",
            ],
        },
    }


def build_corrector_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": CORRECTOR_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]


def normalize_corrector_output(parsed: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    output = dict(parsed)
    output["accept_solver"] = bool(output.get("accept_solver", False))
    output["error_type"] = str(output.get("error_type") or ("none" if output["accept_solver"] else "missing_formula"))
    output["domain"] = str(output.get("domain") or "")
    output["law_family"] = str(output.get("law_family") or "")
    output["target"] = str(output.get("target") or "")

    try:
        output["answer_cardinality"] = int(output.get("answer_cardinality") or 1)
    except (TypeError, ValueError):
        output["answer_cardinality"] = 1
        errors.append("answer_cardinality must be an integer")

    corrected_answer = output.get("corrected_answer")
    if not isinstance(corrected_answer, str) or not corrected_answer.strip():
        if output["accept_solver"]:
            output["corrected_answer"] = ""
        else:
            output["corrected_answer"] = "Unknown"
            errors.append("corrected_answer must be non-empty when accept_solver=false")
    else:
        output["corrected_answer"] = compact_answer_text(corrected_answer)

    premises = output.get("premises", [])
    if premises is None:
        premises = []
    if not isinstance(premises, list):
        premises = [str(premises)]
        errors.append("premises must be a list")
    output["premises"] = [str(item) for item in premises]

    confidence = output.get("confidence", 0.0)
    if isinstance(confidence, (int, float)):
        output["confidence"] = max(0.0, min(1.0, float(confidence)))
    else:
        output["confidence"] = 0.0
        errors.append("confidence must be numeric")
    return output, errors


def corrector_to_physics_response(output: dict[str, Any], baseline_response: dict[str, Any]) -> dict[str, Any]:
    answer = compact_answer_text(str(output.get("corrected_answer") or "Unknown"))
    premises = output.get("premises") or baseline_response.get("premises") or []
    return {
        "answer": answer,
        "explanation": (
            f"Fine-tuned physics corrector selected a correction for {output.get('law_family') or 'this law family'}."
        ),
        "fol": str(baseline_response.get("fol") or ""),
        "cot": numbered_cot(
            [
                "Inspect the question, retrieved physics cards, and deterministic solver result.",
                f"Classify the baseline issue as {output.get('error_type') or 'correction_needed'}.",
                "Return the corrected final answer with the requested unit and answer cardinality.",
            ]
        ),
        "premises": [str(item) for item in premises],
        "confidence": float(output.get("confidence") or 0.0),
    }


@dataclass
class LocalFtCorrector:
    base_model: str
    adapter_path: Path
    max_new_tokens: int = 512

    def __post_init__(self) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover - exercised on Linux training server.
            raise RuntimeError(
                "ft-corrector mode requires torch, transformers, peft, and bitsandbytes. "
                "Install the fine-tune dependencies on the Linux GPU server."
            ) from exc

        compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(self.model, str(self.adapter_path))
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_json(self, payload: dict[str, Any]) -> tuple[dict[str, Any], str, list[str]]:
        messages = build_corrector_messages(payload)
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = (
                f"System: {messages[0]['content']}\n"
                f"User: {messages[1]['content']}\n"
                "Assistant:"
            )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = generated[0][inputs["input_ids"].shape[-1] :]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        parsed = extract_json_object(raw)
        normalized, errors = normalize_corrector_output(parsed)
        return normalized, raw, errors
