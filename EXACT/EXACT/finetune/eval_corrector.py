#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.analyzer import QueryAnalyzer
from model.evaluate import compare_answers, extract_response, iter_jsonl, load_gold, parse_numeric_values, resolve_gold, write_mismatches
from model.physics_ir import build_physics_ir


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def prefix(sample_id: str) -> str:
    match = re.match(r"[A-Za-z]+", sample_id or "")
    return match.group(0) if match else "NOID"


def load_baseline_correct(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    mismatched = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("gold_id"):
                mismatched.add(row["gold_id"])
    return mismatched


def main() -> int:
    configure_stdio()
    parser = argparse.ArgumentParser(description="Evaluate ft-corrector output with law-family metrics.")
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--baseline-mismatches", type=Path, default=None)
    parser.add_argument("--mismatches-out", type=Path, default=None)
    parser.add_argument("--rel-tol", type=float, default=0.03)
    parser.add_argument("--abs-tol", type=float, default=1e-30)
    parser.add_argument("--show", type=int, default=20)
    args = parser.parse_args()

    gold_rows, by_id = load_gold(args.gold)
    baseline_mismatch_ids = load_baseline_correct(args.baseline_mismatches)
    analyzer = QueryAnalyzer()
    total = correct = numeric_total = numeric_correct = text_total = text_correct = 0
    invalid_json = 0
    bad_rows = []
    prefix_counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    law_counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    arbitration_counts: collections.Counter[str] = collections.Counter()
    regression_count = 0
    decision_total = decision_correct = 0
    cardinality_total = cardinality_correct = 0

    for line_no, row in iter_jsonl(args.pred):
        total += 1
        if not row.get("valid_json", True):
            invalid_json += 1
        gold, _ = resolve_gold(row, line_no, gold_rows, by_id)
        if gold is None:
            continue
        response = extract_response(row)
        answer = str(response.get("answer", ""))
        ok, mode, reason = compare_answers(answer, gold, rel_tol=args.rel_tol, abs_tol=args.abs_tol)
        analysis = analyzer.analyze(gold.question)
        ir = build_physics_ir(gold.question, analysis)
        pfx = prefix(gold.sample_id)
        law = f"{ir.domain}/{ir.law_family}"
        prefix_counts[pfx]["total"] += 1
        law_counts[law]["total"] += 1
        if ok:
            correct += 1
            prefix_counts[pfx]["correct"] += 1
            law_counts[law]["correct"] += 1
        elif gold.sample_id not in baseline_mismatch_ids:
            regression_count += 1
        if mode == "numeric":
            numeric_total += 1
            numeric_correct += int(ok)
        else:
            text_total += 1
            text_correct += int(ok)
        arbitration_counts[str(row.get("arbitration") or "")] += 1
        if baseline_mismatch_ids:
            expected_correction = gold.sample_id in baseline_mismatch_ids
            actual_correction = str(row.get("arbitration") or "") == "ft_corrected"
            decision_total += 1
            decision_correct += int(expected_correction == actual_correction or ok)

        pred_count = len(parse_numeric_values(answer))
        gold_count = len(parse_numeric_values(gold.answer, gold.unit))
        if pred_count or gold_count:
            cardinality_total += 1
            cardinality_correct += int(pred_count == gold_count and pred_count > 0)
        if not ok:
            from model.evaluate import EvalResult

            bad_rows.append(
                EvalResult(
                    line_no=line_no,
                    sample_id=str(row.get("sample_id")),
                    gold_id=gold.sample_id,
                    correct=False,
                    mode=mode,
                    pred_answer=answer,
                    gold_answer=gold.answer,
                    question=gold.question,
                    reason=reason,
                )
            )

    law_macro = 0.0
    if law_counts:
        law_macro = sum(counter["correct"] / counter["total"] for counter in law_counts.values() if counter["total"]) / len(law_counts)
    valid_json_rate = 1.0 - (invalid_json / total if total else 0.0)
    overall = correct / total if total else 0.0
    numeric_acc = numeric_correct / numeric_total if numeric_total else 0.0
    decision_acc = decision_correct / decision_total if decision_total else overall
    cardinality_acc = cardinality_correct / cardinality_total if cardinality_total else overall
    no_regression = 1.0 - (regression_count / max(1, total - len(baseline_mismatch_ids)))
    composite = (
        0.35 * numeric_acc
        + 0.20 * law_macro
        + 0.15 * decision_acc
        + 0.10 * cardinality_acc
        + 0.10 * valid_json_rate
        + 0.10 * no_regression
    )

    print(f"rows: {total}")
    print(f"correct: {correct}/{total} = {overall:.4f}")
    if numeric_total:
        print(f"numeric_accuracy: {numeric_correct}/{numeric_total} = {numeric_acc:.4f}")
    if text_total:
        print(f"text_accuracy: {text_correct}/{text_total} = {text_correct / text_total:.4f}")
    print(f"law_family_macro_accuracy: {law_macro:.4f}")
    print(f"correction_decision_accuracy: {decision_correct}/{decision_total} = {decision_acc:.4f}")
    print(f"answer_cardinality_accuracy: {cardinality_correct}/{cardinality_total} = {cardinality_acc:.4f}")
    print(f"valid_json_rate: {valid_json_rate:.4f}")
    print(f"regression_count_on_baseline_correct: {regression_count}")
    print(f"composite_score: {composite:.4f}")
    print("arbitration_counts:")
    for key, value in arbitration_counts.most_common():
        print(f"  {key}: {value}")
    print("prefix_accuracy:")
    for key, counter in sorted(prefix_counts.items()):
        print(f"  {key}: {counter['correct']}/{counter['total']} = {counter['correct'] / counter['total']:.4f}")
    print("top_law_family_accuracy:")
    for key, counter in sorted(law_counts.items(), key=lambda item: (item[1]["correct"] / item[1]["total"], item[0]))[:20]:
        print(f"  {key}: {counter['correct']}/{counter['total']} = {counter['correct'] / counter['total']:.4f}")

    if bad_rows:
        print("first_mismatches:")
        for row in bad_rows[: args.show]:
            print(
                f"  line={row.line_no} sample_id={row.sample_id} gold_id={row.gold_id} "
                f"mode={row.mode} pred={row.pred_answer!r} gold={row.gold_answer!r} reason={row.reason}"
            )
        if args.mismatches_out:
            write_mismatches(args.mismatches_out, bad_rows)
            print(f"mismatches_csv: {args.mismatches_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
