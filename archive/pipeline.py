"""
pipeline.py -- Pipeline Orchestrator: Stage 2 -> Stage 4.

Dieu phoi toan bo quy trinh cho 1 sample:
  Stage 2: Qwen sinh Local Ontology + AST FOL
  Stage 3: Z3 compile & verify
  Stage 4: Feedback loop (Z3 error -> Qwen retry) + Answer extraction
"""

import json
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional

from config import PipelineConfig
from model_loader import QwenModel
from json_parser import safe_json
from z3_compiler import verify_with_z3
from ontology import (
    FORMALIZATION_SYSTEM,
    CORRECTION_SYSTEM,
    ANSWER_SYSTEM,
    PHYSICS_FORMALIZATION_SYSTEM,
    hallucination_check,
)


# ══════════════════════════════════════════════════════════════════
# DATA CLASS: Ket qua cua 1 sample
# ══════════════════════════════════════════════════════════════════

@dataclass
class PipelineResult:
    """Ket qua pipeline cho 1 sample."""
    sample_id:          int
    status:             str   = "pending"
    z3_status:          str   = "pending"
    z3_compiled:        int   = 0
    z3_total:           int   = 0
    z3_attempts:        int   = 0
    z3_errors:          list  = field(default_factory=list)
    local_ontology:     list  = field(default_factory=list)
    premises_ast:       list  = field(default_factory=list)
    hallucination_warn: list  = field(default_factory=list)
    predicted_answers:  list  = field(default_factory=list)
    ground_truth:       list  = field(default_factory=list)
    correct_count:      int   = 0
    total_questions:    int   = 0
    time_sec:           float = 0.0
    error_log:          list  = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════
# STAGE 2: Qwen sinh Local Ontology + AST
# ══════════════════════════════════════════════════════════════════

def run_formalization(qwen: QwenModel, premises_nl: list,
                      is_physics: bool = False) -> dict:
    """Goi Qwen 2 buoc:
      Buoc 1: NER + Local Ontology
      Buoc 2: AST FOL de quy (chi dung Global + Local Ontology)

    Returns:
        dict { step1_local_ontology: [...], step2_premises_ast: [...] }
    """
    numbered = "\n".join(
        f"Premise {i + 1}: {p}" for i, p in enumerate(premises_nl)
    )

    user_msg = (
        "Hay hinh thuc hoa cac tien de sau theo dung quy trinh 2 buoc.\n\n"
        + numbered
        + "\n\nNho:\n"
        "  Buoc 1: khai bao Local Ontology -- tat ca khai niem quan trong trong van ban\n"
        "  Buoc 2: dung DUNG ten + arity Predicate tu Buoc 1, sinh cay AST JSON de quy cho tung premise\n"
        "  Chi tra ve JSON thuan tuy -- khong co text, khong co markdown."
    )

    system = PHYSICS_FORMALIZATION_SYSTEM if is_physics else FORMALIZATION_SYSTEM
    raw = qwen.generate(system, user_msg)
    return safe_json(raw)


# ══════════════════════════════════════════════════════════════════
# STAGE 4a: Feedback Loop (Z3 -> Qwen)
# ══════════════════════════════════════════════════════════════════

def run_correction(
    qwen: QwenModel,
    premises_nl: list,
    prev_formalization: Optional[dict],
    z3_info: Optional[dict],
    hall_warnings: list,
) -> dict:
    """Gui thong bao loi Z3 ve Qwen de sua lai.

    BUG FIX #1b: None-guard cho prev_formalization va z3_info.

    Returns:
        New formalization dict.
    """
    numbered = "\n".join(
        f"Premise {i + 1}: {p}" for i, p in enumerate(premises_nl)
    )

    safe_z3 = z3_info or {}
    safe_prev = prev_formalization or {}

    z3_errors = "\n".join(safe_z3.get("errors", [])) or "(khong co loi compile cu the)"
    hall_errs = "\n".join(hall_warnings) if hall_warnings else "(khong co)"
    prev_local = json.dumps(
        safe_prev.get("step1_local_ontology", []),
        ensure_ascii=False,
        indent=2,
    )

    correction_user = (
        "He thong Z3 da phat hien loi khi compile cay AST cua ban.\n\n"
        "===================================================\n"
        "THONG TIN LOI TU Z3\n"
        "===================================================\n"
        f'Z3 status: {safe_z3.get("status", "N/A")}\n'
        f'So premise compile duoc: {safe_z3.get("compiled_count", 0)} '
        f'/ {safe_z3.get("total_count", 0)}\n\n'
        f"Loi compile chi tiet:\n{z3_errors}\n\n"
        f"Loi Hallucination (Predicate khong khai bao):\n{hall_errs}\n\n"
        "===================================================\n"
        "LOCAL ONTOLOGY LAN TRUOC (de tham khao)\n"
        "===================================================\n"
        f"{prev_local}\n\n"
        "===================================================\n"
        "PREMISES GOC\n"
        "===================================================\n"
        f"{numbered}\n\n"
        "Hay sua lai TOAN BO (Buoc 1 + Buoc 2) de khong con loi.\n"
        "Chi tra ve JSON thuan tuy."
    )

    raw = qwen.generate(CORRECTION_SYSTEM, correction_user)
    return safe_json(raw)


# ══════════════════════════════════════════════════════════════════
# STAGE 4b: Answer Extraction
# ══════════════════════════════════════════════════════════════════

def extract_answers(
    qwen: QwenModel,
    premises_nl: list,
    fol_context: list,
    questions: list,
    config: PipelineConfig,
) -> list:
    """Dung Qwen tra loi cau hoi dua tren FOL da xac minh.

    Args:
        qwen: Loaded QwenModel.
        premises_nl: Original premises.
        fol_context: FOL representation strings.
        questions: List of question strings.
        config: Pipeline config (for ans_max_tokens).

    Returns:
        List of { question_id, answer, reasoning }.
    """
    p_text = "\n".join(f"P{i + 1}: {p}" for i, p in enumerate(premises_nl))
    fol_text = "\n".join(f"FOL P{i + 1}: {f}" for i, f in enumerate(fol_context))

    answers = []
    for i, q in enumerate(questions):
        user_msg = (
            "## Tien de (Natural Language):\n"
            f"{p_text}\n\n"
            "## Tien de (FOL da xac minh qua Z3):\n"
            f"{fol_text}\n\n"
            f"## Cau hoi {i + 1}:\n"
            f"{q}\n\n"
            "Tra loi JSON thuan tuy."
        )

        raw = qwen.generate(ANSWER_SYSTEM, user_msg, max_new_tokens=config.ans_max_tokens)
        try:
            ans = safe_json(raw)
            answers.append({
                "question_id": i,
                "answer": ans.get("answer", "Unknown"),
                "reasoning": ans.get("reasoning", ""),
            })
        except Exception:
            answers.append({
                "question_id": i,
                "answer": "Unknown",
                "reasoning": raw[:200],
            })
    return answers


# ══════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR (1 sample)
# ══════════════════════════════════════════════════════════════════

def run_pipeline(
    idx: int,
    sample: dict,
    qwen: QwenModel,
    config: PipelineConfig,
    is_physics: bool = False,
) -> PipelineResult:
    """Chay toan bo pipeline cho 1 sample.

    Flow:
      1. Qwen formalization (Stage 2)
      2. Z3 compile & verify (Stage 3)
      3. If error -> feedback loop -> Qwen correction (Stage 4a)
      4. Repeat until success or MAX_RETRIES exhausted
      5. Extract answers (Stage 4b) -- LUON chay, ke ca khi Z3 that bai
      6. Score against ground truth

    Args:
        idx: Sample index.
        sample: Dict voi keys premises-NL, questions, answers.
        qwen: Loaded QwenModel.
        config: Pipeline config.
        is_physics: True if processing physics dataset.

    Returns:
        PipelineResult.
    """
    premises_nl = sample.get("premises-NL", sample.get("premises", []))
    questions = sample.get("questions", [])
    gt_answers = sample.get("answers", [])

    result = PipelineResult(
        sample_id=idx,
        ground_truth=gt_answers,
        total_questions=len(questions),
    )
    t0 = time.time()
    print(
        f"  [Sample {idx:02d}] {len(premises_nl)} premises, {len(questions)} Q",
        end="",
        flush=True,
    )

    formalization = None
    z3_info = None

    # ── Vong lap Qwen -> Z3 -> Feedback ──────────────────────────
    for attempt in range(1, config.max_retries + 1):
        result.z3_attempts = attempt
        try:
            # BUG FIX #1: Neu formalization hoac z3_info la None
            # (vi attempt truoc that bai), LUON goi run_formalization lai.
            if attempt == 1 or formalization is None or z3_info is None:
                if attempt > 1:
                    print(f" [retry {attempt - 1}]", end="", flush=True)
                formalization = run_formalization(qwen, premises_nl, is_physics)
            else:
                # Stage 4a: co du du lieu -> Qwen sua loi theo feedback Z3
                print(f" [retry {attempt - 1}]", end="", flush=True)
                formalization = run_correction(
                    qwen, premises_nl, formalization, z3_info,
                    result.hallucination_warn,
                )

            local_onto = formalization.get("step1_local_ontology", [])
            premises_ast = formalization.get("step2_premises_ast", [])

            if not premises_ast:
                raise ValueError("step2_premises_ast rong -- Qwen chua sinh AST")

            # Hallucination check
            hw = hallucination_check(local_onto, premises_ast)

            # Stage 3: Z3 compile & verify
            z3_info = verify_with_z3(premises_ast)

            # Luu ket qua tam
            result.local_ontology = local_onto
            result.premises_ast = premises_ast
            result.hallucination_warn = hw
            result.z3_status = z3_info["status"]
            result.z3_errors = z3_info.get("errors", [])
            result.z3_compiled = z3_info.get("compiled_count", 0)
            result.z3_total = z3_info.get("total_count", 0)

            if z3_info["status"] != "compile_error":
                break  # thanh cong (sat / unsat / unknown / solver_error)

        except Exception as e:
            result.error_log.append(
                f"Attempt {attempt}: {traceback.format_exc()[-500:]}"
            )
            # BUG FIX #5: KHONG return ngay khi het retries.
            if attempt == config.max_retries:
                print(" [all retries failed]", end="", flush=True)
                break

    # ── Fallback premises_ast ─────────────────────────────────────
    if not result.premises_ast:
        result.premises_ast = [
            {"premise_id": i, "source_nl": p, "ast": {}}
            for i, p in enumerate(premises_nl)
        ]
        if result.z3_status == "pending":
            result.z3_status = "no_ast"

    # ── Stage 4b: Tra loi cau hoi (LUON chay) ────────────────────
    fol_ctx = [item.get("source_nl", "") for item in result.premises_ast]
    ans_results = extract_answers(qwen, premises_nl, fol_ctx, questions, config)
    result.predicted_answers = ans_results

    # ── Danh gia ──────────────────────────────────────────────────
    correct = sum(
        1
        for i, ar in enumerate(ans_results)
        if i < len(gt_answers)
        and str(ar["answer"]).strip().upper() == str(gt_answers[i]).strip().upper()
    )
    result.correct_count = correct

    # Xac dinh status cuoi
    if result.z3_status in ("sat", "unsat", "unknown"):
        result.status = "success"
    elif result.z3_compiled > 0:
        result.status = "partial"
    else:
        result.status = "failed"

    result.time_sec = round(time.time() - t0, 2)

    badge = {
        "sat": "sat OK",
        "unsat": "unsat(!)",
        "unknown": "unknown?",
        "compile_error": "ERR_COMPILE",
        "solver_error": "ERR_SOLVER",
        "no_ast": "NO_AST",
    }.get(result.z3_status, result.z3_status)

    print(
        f" | {badge} | {correct}/{len(questions)} correct | {result.time_sec}s"
    )
    return result
