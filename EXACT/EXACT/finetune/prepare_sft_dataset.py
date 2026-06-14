#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.analyzer import QueryAnalyzer
from model.evaluate import GoldRow, compare_answers, extract_response, gold_answer_text, load_gold
from model.ft_corrector import CORRECTOR_SYSTEM_PROMPT, build_corrector_payload
from model.kb import KnowledgeBase
from model.physics_ir import build_physics_ir
from model.solver import DeterministicPhysicsSolver


UNIT_FACTORS = {
    "": 1.0,
    "v/m": 1.0,
    "n/c": 1.0,
    "kv/m": 1.0e3,
    "n": 1.0,
    "mn": 1.0e-3,
    "j": 1.0,
    "mj": 1.0e-3,
    "uj": 1.0e-6,
    "nj": 1.0e-9,
    "c": 1.0,
    "uc": 1.0e-6,
    "mc": 1.0e-3,
    "v": 1.0,
    "ohm": 1.0,
    "hz": 1.0,
    "a": 1.0,
    "ma": 1.0e-3,
    "%": 0.01,
}


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_mismatches(path: Path) -> dict[int, dict[str, str]]:
    mismatches: dict[int, dict[str, str]] = {}
    if not path.exists():
        return mismatches
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                mismatches[int(row["line_no"])] = row
            except (KeyError, ValueError):
                continue
    return mismatches


def normalize_unit(unit: str) -> str:
    value = str(unit or "").strip().lower()
    value = value.replace("μ", "u").replace("µ", "u").replace("ω", "ohm").replace("ohms", "ohm")
    value = value.replace(" ", "")
    return value


def first_numeric_value(text: str, default_unit: str = "") -> float | None:
    value = str(text or "")
    value = value.replace("×", "x").replace("*", "x").replace("−", "-")
    frac = re.search(r"([-+]?\d+(?:\.\d+)?)\s*/\s*([-+]?\d+(?:\.\d+)?)", value)
    if frac:
        number = float(frac.group(1)) / float(frac.group(2))
        unit_match = re.search(r"(kv/m|v/m|n/c|mn|mj|uj|nj|uc|mc|ohm|hz|[nvjac%])\b", value[frac.end() :].lower())
        unit = normalize_unit(unit_match.group(1) if unit_match else default_unit)
        return number * UNIT_FACTORS.get(unit, 1.0)
    sci = re.search(r"([-+]?\d+(?:\.\d+)?)\s*x\s*10\s*\^?\s*\{?\s*([-+]?\d+)\s*\}?", value, re.IGNORECASE)
    if sci:
        number = float(sci.group(1)) * (10.0 ** int(sci.group(2)))
        unit_match = re.search(r"(kv/m|v/m|n/c|mn|mj|uj|nj|uc|mc|ohm|hz|[nvjac%])\b", value[sci.end() :].lower())
        unit = normalize_unit(unit_match.group(1) if unit_match else default_unit)
        return number * UNIT_FACTORS.get(unit, 1.0)
    num = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    if not num:
        return None
    unit_match = re.search(r"(kv/m|v/m|n/c|mn|mj|uj|nj|uc|mc|ohm|hz|[nvjac%])\b", value[num.end() :].lower())
    unit = normalize_unit(unit_match.group(1) if unit_match else default_unit)
    return float(num.group(0)) * UNIT_FACTORS.get(unit, 1.0)


def is_lenient_equivalent(pred_answer: str, gold: GoldRow) -> bool:
    ok, _, _ = compare_answers(pred_answer, gold, rel_tol=0.04, abs_tol=1e-10)
    if ok:
        return True
    pred_value = first_numeric_value(pred_answer)
    gold_value = first_numeric_value(gold.answer, gold.unit)
    if pred_value is None or gold_value is None:
        return False
    return math.isclose(pred_value, gold_value, rel_tol=0.05, abs_tol=1e-10)


def classify_error(mismatch: dict[str, str] | None, ir_law_family: str, baseline_answer: str, gold: GoldRow) -> str:
    if mismatch is None:
        return "none"
    reason = (mismatch.get("reason") or "").lower()
    mode = (mismatch.get("mode") or "").lower()
    if mode == "text" or reason.startswith("text"):
        return "conceptual_text"
    if "numeric_count" in reason:
        return "wrong_cardinality"
    if any(unit in reason for unit in ["kv/m", "mn", "nj", "uc", "mc"]):
        return "wrong_unit"
    if ir_law_family in {"distributed_charge", "segmented_phasor", "uniform_field_motion", "zero_field_inverse"}:
        return "wrong_law"
    if first_numeric_value(baseline_answer) is not None and first_numeric_value(gold.answer, gold.unit) is not None:
        return "arithmetic"
    return "wrong_target"


def answer_cardinality(answer: str) -> int:
    text = str(answer or "")
    if ";" in text:
        return len([part for part in text.split(";") if part.strip()])
    if " and " in text.lower():
        return len([part for part in re.split(r"\band\b", text, flags=re.IGNORECASE) if part.strip()])
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?(?:\s*(?:x|\*)\s*10\s*\^?\s*\{?[-+]?\d+\}?)?", text)
    return max(1, len(numbers)) if numbers else 1


def prefix(sample_id: str) -> str:
    match = re.match(r"[A-Za-z]+", sample_id or "")
    return match.group(0) if match else "NOID"


def group_key_for(row: dict[str, Any]) -> str:
    parts = [
        row["meta"]["prefix"],
        row["assistant"]["domain"],
        row["assistant"]["law_family"],
        row["meta"]["route"],
        row["assistant"]["target"],
        ",".join(row["meta"].get("geometry_tags", [])),
    ]
    return "|".join(str(part) for part in parts)


def split_groups(rows: list[dict[str, Any]], seed: int) -> dict[str, str]:
    grouped: dict[str, list[int]] = collections.defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[group_key_for(row)].append(idx)
    groups = list(grouped)
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=lambda key: hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest())

    total = len(rows)
    train_limit = int(total * 0.70)
    valid_limit = int(total * 0.85)
    assignment: dict[str, str] = {}
    count = 0
    for key in groups:
        size = len(grouped[key])
        if count < train_limit:
            split = "train"
        elif count < valid_limit:
            split = "valid"
        else:
            split = "test"
        assignment[key] = split
        count += size
    return assignment


def make_sft_row(row: dict[str, Any]) -> dict[str, Any]:
    user_payload = row["user_payload"]
    assistant = row["assistant"]
    return {
        "messages": [
            {"role": "system", "content": CORRECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, sort_keys=True)},
            {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False, sort_keys=True)},
        ],
        "meta": row["meta"],
    }


def main() -> int:
    configure_stdio()
    parser = argparse.ArgumentParser(description="Build grouped SFT data for the EXACT Type 2 physics corrector.")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--mismatches", type=Path, required=True)
    parser.add_argument("--kb-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--formula-top-k", type=int, default=4)
    parser.add_argument("--geometry-top-k", type=int, default=3)
    parser.add_argument("--example-top-k", type=int, default=3)
    args = parser.parse_args()

    gold_rows, _ = load_gold(args.gold)
    pred_rows = load_jsonl(args.pred)
    mismatches = load_mismatches(args.mismatches)
    if len(pred_rows) != len(gold_rows):
        raise ValueError(f"Prediction/gold length mismatch: pred={len(pred_rows)} gold={len(gold_rows)}")

    analyzer = QueryAnalyzer()
    solver = DeterministicPhysicsSolver()
    kb = KnowledgeBase(args.kb_root)
    rows: list[dict[str, Any]] = []

    for idx, (pred, gold) in enumerate(zip(pred_rows, gold_rows, strict=True), start=1):
        question = gold.question
        analysis = analyzer.analyze(question)
        ir = build_physics_ir(question, analysis)
        retrieval = kb.retrieve(
            analysis,
            formula_top_k=args.formula_top_k,
            geometry_top_k=args.geometry_top_k,
            example_top_k=args.example_top_k,
        )
        solver_result = solver.solve(question)
        baseline_response = extract_response(pred)
        baseline_answer = str(baseline_response.get("answer", "Unknown"))
        mismatch = mismatches.get(idx)
        accept_solver = mismatch is None or is_lenient_equivalent(baseline_answer, gold)
        corrected_answer = baseline_answer if accept_solver else gold_answer_text(gold)
        error_type = "none" if accept_solver else classify_error(mismatch, ir.law_family, baseline_answer, gold)
        assistant = {
            "accept_solver": accept_solver,
            "error_type": error_type,
            "domain": ir.domain,
            "law_family": ir.law_family,
            "target": ir.target,
            "answer_cardinality": answer_cardinality(corrected_answer),
            "corrected_answer": corrected_answer,
            "premises": list(dict.fromkeys([*(solver_result.premises or []), *[card.id for card in retrieval.formula_cards[:2]]])),
            "confidence": 0.92 if accept_solver else 0.86,
        }
        payload = build_corrector_payload(
            question=question,
            analysis=analysis,
            retrieval=retrieval,
            solver_result=solver_result,
            baseline_response=baseline_response,
        )
        rows.append(
            {
                "user_payload": payload,
                "assistant": assistant,
                "meta": {
                    "row_number": idx,
                    "gold_id": gold.sample_id,
                    "prefix": prefix(gold.sample_id),
                    "route": str(pred.get("route") or solver_result.route),
                    "group_key": "",
                    "is_mismatch": mismatch is not None,
                    "accepted_as_equivalent": mismatch is not None and accept_solver,
                    "geometry_tags": analysis.geometry_tags,
                },
            }
        )

    for row in rows:
        row["meta"]["group_key"] = group_key_for(row)
    split_assignment = split_groups(rows, args.seed)
    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    for row in rows:
        split = split_assignment[row["meta"]["group_key"]]
        split_rows[split].append(make_sft_row(row))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, items in split_rows.items():
        with (args.out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    manifest = {
        "seed": args.seed,
        "source_files": {
            "gold": str(args.gold),
            "pred": str(args.pred),
            "mismatches": str(args.mismatches),
            "kb_root": str(args.kb_root),
        },
        "counts": {split: len(items) for split, items in split_rows.items()},
        "total": len(rows),
        "mismatch_rows": len(mismatches),
        "accepted_equivalent_mismatches": sum(1 for row in rows if row["meta"]["accepted_as_equivalent"]),
        "group_count": len(split_assignment),
        "splits_by_group": split_assignment,
        "schema": {
            "messages": "OpenAI-style chat messages; user content is a JSON string; assistant content is a JSON string.",
            "meta": "Training metadata only. Do not include sample id in prompts.",
        },
    }
    (args.out_dir / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("counts", "total", "mismatch_rows", "accepted_equivalent_mismatches", "group_count")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
