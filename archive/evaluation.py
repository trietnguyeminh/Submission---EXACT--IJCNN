"""
evaluation.py -- Stage 5: Evaluation & Export.

Tinh toan metrics, in summary table, va luu ket qua ra JSON.
"""

import json
import time
from pathlib import Path

from pipeline import PipelineResult


def evaluate(results: list) -> dict:
    """Tinh toan metrics tong hop tu list PipelineResult.

    Returns:
        Dict voi cac metrics: accuracy, z3 breakdown, hallucination stats, etc.
    """
    n = len(results)
    if n == 0:
        return {}

    total_q = sum(r.total_questions for r in results)
    total_ok = sum(r.correct_count for r in results)

    status_ct = {"success": 0, "partial": 0, "failed": 0}
    z3_ct = {
        "sat": 0,
        "unsat": 0,
        "unknown": 0,
        "compile_error": 0,
        "solver_error": 0,
        "other": 0,
    }

    for r in results:
        status_ct[r.status] = status_ct.get(r.status, 0) + 1
        key = r.z3_status if r.z3_status in z3_ct else "other"
        z3_ct[key] += 1

    hall_total = sum(len(r.hallucination_warn) for r in results)
    avg_retries = sum(r.z3_attempts for r in results) / n
    avg_time = sum(r.time_sec for r in results) / n
    avg_comp = sum(r.z3_compiled for r in results) / n
    avg_tot_p = sum(r.z3_total for r in results) / n

    return {
        "n_samples": n,
        "total_questions": total_q,
        "total_correct": total_ok,
        "accuracy": round(total_ok / total_q, 4) if total_q else 0,
        "status_breakdown": status_ct,
        "z3_breakdown": z3_ct,
        "hallucination_warnings": hall_total,
        "avg_z3_retries": round(avg_retries, 2),
        "avg_time_sec": round(avg_time, 2),
        "avg_compiled_pct": (
            round(avg_comp / avg_tot_p * 100, 1) if avg_tot_p else 0
        ),
    }


def print_summary(metrics: dict, model_id: str, quantization: str):
    """In bang tong hop ket qua ra console."""
    W = 58
    print("=" * W)
    print("  NEURO-SYMBOLIC PIPELINE -- EVALUATION SUMMARY")
    print(f"  Model: {model_id} ({quantization})")
    print("=" * W)
    print(f'  Samples evaluated  : {metrics["n_samples"]}')
    print(f'  Total questions    : {metrics["total_questions"]}')
    print(f'  Correct answers    : {metrics["total_correct"]}')
    print(f'  Accuracy           : {metrics["accuracy"]:.1%}')
    print("-" * W)
    print("  Pipeline Status:")
    for k, v in metrics["status_breakdown"].items():
        print(f"    {k:14}: {v:3d}  {'#' * v}")
    print("-" * W)
    print("  Z3 Verification:")
    for k, v in metrics["z3_breakdown"].items():
        if v > 0:
            print(f"    {k:16}: {v:3d}  {'#' * v}")
    print("-" * W)
    print(f'  Hallucination warns: {metrics["hallucination_warnings"]}')
    print(f'  Avg Z3 retries     : {metrics["avg_z3_retries"]}')
    print(f'  Avg compile rate   : {metrics["avg_compiled_pct"]}%')
    print(f'  Avg time / sample  : {metrics["avg_time_sec"]}s')
    print("=" * W)


def print_per_sample(results: list):
    """In chi tiet tung sample ra console."""
    header = (
        f"{'ID':>3} | {'Status':>8} | {'Z3':>13} | "
        f"{'Corr':>6} | {'Retry':>5} | {'Time':>6} | Hall | Predicted"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        pred_ans = [a["answer"] for a in r.predicted_answers]
        paired = "  ".join(
            f'{"v" if str(p).upper() == str(g).upper() else "x"}{p}(gt:{g})'
            for p, g in zip(pred_ans, r.ground_truth)
        )
        hall = f"W{len(r.hallucination_warn)}" if r.hallucination_warn else "ok"
        print(
            f"{r.sample_id:>3} | {r.status:>8} | {r.z3_status:>13} | "
            f"{r.correct_count}/{r.total_questions:>4} | {r.z3_attempts:>5} | "
            f"{r.time_sec:>5.1f}s | {hall:>4} | {paired}"
        )


def result_to_dict(r: PipelineResult) -> dict:
    """Chuyen PipelineResult -> dict de luu JSON."""
    return {
        "sample_id": r.sample_id,
        "status": r.status,
        "z3_status": r.z3_status,
        "z3_compiled": r.z3_compiled,
        "z3_total": r.z3_total,
        "z3_attempts": r.z3_attempts,
        "z3_errors": r.z3_errors[:3],
        "hallucination_warns": r.hallucination_warn,
        "local_ontology": r.local_ontology,
        "correct_count": r.correct_count,
        "total_questions": r.total_questions,
        "predicted_answers": [a["answer"] for a in r.predicted_answers],
        "ground_truth": r.ground_truth,
        "per_question": [
            {
                "q_id": a["question_id"],
                "predicted": a["answer"],
                "gt": (
                    r.ground_truth[a["question_id"]]
                    if a["question_id"] < len(r.ground_truth)
                    else "?"
                ),
                "correct": (
                    str(a["answer"]).upper()
                    == str(r.ground_truth[a["question_id"]]).upper()
                    if a["question_id"] < len(r.ground_truth)
                    else False
                ),
                "reasoning": a.get("reasoning", ""),
            }
            for a in r.predicted_answers
        ],
        "time_sec": r.time_sec,
        "error_log": r.error_log[-1:],
    }


def export_results(
    results: list,
    metrics: dict,
    config,
    output_path: str,
):
    """Luu toan bo ket qua ra file JSON.

    Args:
        results: List PipelineResult.
        metrics: Dict tu evaluate().
        config: PipelineConfig.
        output_path: Duong dan file output.
    """
    output_data = {
        "meta": {
            "model": config.model_id,
            "quantization": config.quantization,
            "n_samples": config.n_samples,
            "max_retries": config.max_retries,
            "dataset": config.dataset_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": metrics,
        "per_sample": [result_to_dict(r) for r in results],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    file_size = Path(output_path).stat().st_size / 1024
    print(f"\nKet qua luu tai: {output_path}")
    print(f"  Dung luong: {file_size:.1f} KB")
    print(
        f'  Final Accuracy: {metrics["accuracy"]:.1%}  '
        f'({metrics["total_correct"]}/{metrics["total_questions"]})'
    )
