#!/usr/bin/env python3
"""Evaluate EXACT Type 2 physics JSONL outputs against the local gold CSV.

The evaluator accepts the audit JSONL produced by `python -m model.cli`.
It can also evaluate bare response JSONL lines as long as each line contains an
`answer` field. Gold matching works by `sample_id` when it is a dataset id, by
1-based row order when `sample_id` is numeric, and finally by `item_index`.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPERSCRIPT_TRANSLATION = str.maketrans(
    {
        "\u2070": "0",
        "\u00b9": "1",
        "\u00b2": "2",
        "\u00b3": "3",
        "\u2074": "4",
        "\u2075": "5",
        "\u2076": "6",
        "\u2077": "7",
        "\u2078": "8",
        "\u2079": "9",
        "\u207b": "-",
        "\u2080": "0",
        "\u2081": "1",
        "\u2082": "2",
        "\u2083": "3",
        "\u2084": "4",
        "\u2085": "5",
        "\u2086": "6",
        "\u2087": "7",
        "\u2088": "8",
        "\u2089": "9",
    }
)

NUMBER_RE = re.compile(
    r"(?<![A-Za-z])[-+]?(?:10\s*\^\s*\{?[-+]?\d+\}?|\d+(?:\.\d+)?(?:\s*(?:x|\*|X)\s*10\s*\^\s*\{?[-+]?\d+\}?)?)"
)
SQRT_NUMBER_RE = re.compile(
    r"(?:(?P<coef>[-+]?\d+(?:\.\d+)?)\s*(?:x|\*|X)?\s*)?"
    r"sqrt\s*\(?\s*(?P<rad>\d+(?:\.\d+)?)\s*\)?"
    r"\s*(?:(?:x|\*|X)\s*10\s*\^\s*\{?(?P<exp>[-+]?\d+)\}?)?"
)

UNIT_FACTORS: dict[str, float] = {
    "": 1.0,
    "-": 1.0,
    "—": 1.0,
    "F": 1.0,
    "uF": 1.0e-6,
    "microF": 1.0e-6,
    "mF": 1.0e-3,
    "nF": 1.0e-9,
    "pF": 1.0e-12,
    "C": 1.0,
    "uC": 1.0e-6,
    "microC": 1.0e-6,
    "mC": 1.0e-3,
    "nC": 1.0e-9,
    "V": 1.0,
    "J": 1.0,
    "mJ": 1.0e-3,
    "uJ": 1.0e-6,
    "microJ": 1.0e-6,
    "nJ": 1.0e-9,
    "H": 1.0,
    "mH": 1.0e-3,
    "A": 1.0,
    "mA": 1.0e-3,
    "ohm": 1.0,
    "Hz": 1.0,
    "kHz": 1.0e3,
    "T": 1.0,
    "mT": 1.0e-3,
    "Wb": 1.0,
    "uWb": 1.0e-6,
    "W": 1.0,
    "V/m": 1.0,
    "N/C": 1.0,
    "N": 1.0,
    "%": 0.01,
}

UNIT_PATTERNS = sorted(UNIT_FACTORS.keys(), key=len, reverse=True)


@dataclass
class GoldRow:
    row_number: int
    sample_id: str
    question: str
    answer: str
    unit: str


@dataclass
class EvalResult:
    line_no: int
    sample_id: str
    gold_id: str
    correct: bool
    mode: str
    pred_answer: str
    gold_answer: str
    question: str
    reason: str


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def normalize_text(text: Any) -> str:
    value = str(text or "").translate(SUPERSCRIPT_TRANSLATION)
    replacements = {
        "\u00d7": "x",
        "\u2212": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u03bc": "u",
        "\u00b5": "u",
        "\u03a9": "ohm",
        "\u221a": "sqrt",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def parse_number(token: str) -> float:
    legacy = normalize_text(token)
    legacy = re.sub(r"(\d(?:\.\d+)?)\s+\.\s*10\s*\^?\{?([-+]?\d+)\}?", r"\1x10^\2", legacy)
    value = legacy.replace(" ", "")
    bare_power = re.fullmatch(r"([-+]?)10\^\{?([-+]?\d+)\}?", value)
    if bare_power:
        sign = -1.0 if bare_power.group(1) == "-" else 1.0
        return sign * (10.0 ** int(bare_power.group(2)))
    match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)(?:x|\*|X)10\^?\{?([-+]?\d+)\}?", value)
    if match:
        return float(match.group(1)) * (10.0 ** int(match.group(2)))
    return float(value)


def parse_sqrt_number(match: re.Match[str]) -> float:
    coef = float(match.group("coef") or "1")
    radicand = float(match.group("rad"))
    exponent = int(match.group("exp") or "0")
    return coef * math.sqrt(radicand) * (10.0 ** exponent)


def remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in spans:
        for idx in range(start, end):
            chars[idx] = " "
    return "".join(chars)


def strip_symbolic_labels(text: str) -> str:
    return re.sub(r"\b(?:q|Q|F|E|U|I|R|C|L|W|X)[A-Za-z]*\d+\b", lambda match: re.sub(r"\d+", "", match.group(0)), text)


def canonical_unit(unit: str) -> str:
    value = normalize_text(unit)
    value = value.replace("Ohm", "ohm").replace("ohms", "ohm")
    value = value.strip()
    if value == "N/C":
        return "V/m"
    return value


def normalize_answer_text(text: str) -> str:
    value = normalize_text(text).lower()
    value = value.replace("—", "-")
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .")


def split_default_units(unit_field: str) -> list[str]:
    unit = normalize_text(unit_field)
    if not unit or unit in {"-", "—"}:
        return []
    return [canonical_unit(part.strip()) for part in unit.split(";")]


def detect_unit_after(text: str, end_pos: int) -> str:
    tail = text[end_pos : end_pos + 24].strip()
    for unit in UNIT_PATTERNS:
        if not unit or unit in {"-", "—"}:
            continue
        if tail.startswith(unit):
            return canonical_unit(unit)
    return ""


def split_answer_segments(answer: str) -> list[str]:
    normalized = normalize_text(answer)
    if ";" in normalized:
        return [part.strip() for part in normalized.split(";") if part.strip()]
    if " and " in normalized and len(NUMBER_RE.findall(normalized)) > 1:
        return [part.strip() for part in normalized.split(" and ") if part.strip()]
    return [normalized]


def parse_numeric_values(answer: str, unit_field: str = "") -> list[tuple[float, str]]:
    default_units = split_default_units(unit_field)
    values: list[tuple[float, str]] = []
    segments = split_answer_segments(answer)
    for seg_idx, segment in enumerate(segments):
        if "=" in segment:
            segment = segment.rsplit("=", 1)[-1]
        segment = strip_symbolic_labels(segment)
        segment = re.sub(r"(\d(?:\.\d+)?)\s+\.\s*10\s*\^?\{?([-+]?\d+)\}?", r"\1x10^\2", segment)
        sqrt_spans: list[tuple[int, int]] = []
        for sqrt_match in SQRT_NUMBER_RE.finditer(segment):
            value = parse_sqrt_number(sqrt_match)
            unit = detect_unit_after(segment, sqrt_match.end())
            if not unit and seg_idx < len(default_units):
                unit = default_units[seg_idx]
            factor = UNIT_FACTORS.get(unit, 1.0)
            values.append((value * factor, unit))
            sqrt_spans.append(sqrt_match.span())
        if sqrt_spans:
            segment = remove_spans(segment, sqrt_spans)
        matches = list(NUMBER_RE.finditer(segment))
        for match_idx, match in enumerate(matches):
            try:
                value = parse_number(match.group(0))
            except ValueError:
                continue
            unit = detect_unit_after(segment, match.end())
            if not unit:
                if seg_idx < len(default_units):
                    unit = default_units[seg_idx]
                elif len(default_units) == len(matches):
                    unit = default_units[match_idx]
                elif len(default_units) == 1 and len(matches) == 1:
                    unit = default_units[0]
                else:
                    unit = ""
            factor = UNIT_FACTORS.get(unit, 1.0)
            values.append((value * factor, unit))
    return values


def gold_answer_text(row: GoldRow) -> str:
    if row.unit and row.unit not in {"-", "—"}:
        return f"{row.answer} {row.unit}"
    return row.answer


def load_gold(path: Path) -> tuple[list[GoldRow], dict[str, GoldRow]]:
    rows: list[GoldRow] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "question", "answer", "unit"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Gold CSV is missing columns: {sorted(missing)}")
        for idx, row in enumerate(reader, start=1):
            rows.append(
                GoldRow(
                    row_number=idx,
                    sample_id=str(row["id"]),
                    question=row["question"],
                    answer=row["answer"],
                    unit=row["unit"],
                )
            )
    by_id = {row.sample_id: row for row in rows}
    return rows, by_id


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.strip():
                yield line_no, json.loads(line)


def resolve_gold(row: dict[str, Any], line_no: int, gold_rows: list[GoldRow], by_id: dict[str, GoldRow]) -> tuple[GoldRow | None, str]:
    sample_id = row.get("sample_id")
    if sample_id is not None:
        sample_key = str(sample_id)
        if sample_key in by_id:
            return by_id[sample_key], "sample_id"
        if sample_key.isdigit():
            index = int(sample_key)
            if 1 <= index <= len(gold_rows):
                return gold_rows[index - 1], "sample_id_row_number"
    item_index = row.get("item_index")
    if isinstance(item_index, int) and 0 <= item_index < len(gold_rows):
        return gold_rows[item_index], "item_index"
    if 1 <= line_no <= len(gold_rows):
        return gold_rows[line_no - 1], "line_order"
    return None, "unmatched"


def extract_response(row: dict[str, Any]) -> dict[str, Any]:
    response = row.get("response")
    if isinstance(response, dict):
        return response
    if "answer" in row:
        return row
    return {}


def compare_answers(pred_answer: str, gold: GoldRow, rel_tol: float, abs_tol: float) -> tuple[bool, str, str]:
    pred_values = parse_numeric_values(pred_answer)
    gold_values = parse_numeric_values(gold.answer, gold.unit)
    if pred_values or gold_values:
        if len(pred_values) != len(gold_values) or not pred_values:
            return False, "numeric", f"numeric_count pred={len(pred_values)} gold={len(gold_values)}"
        for idx, ((pred_value, pred_unit), (gold_value, gold_unit)) in enumerate(zip(pred_values, gold_values), start=1):
            tolerance_ok = math.isclose(pred_value, gold_value, rel_tol=rel_tol, abs_tol=abs_tol)
            if not tolerance_ok:
                return (
                    False,
                    "numeric",
                    f"value_{idx} pred={pred_value:g}({pred_unit}) gold={gold_value:g}({gold_unit})",
                )
        return True, "numeric", "ok"

    pred_text = normalize_answer_text(pred_answer)
    gold_text = normalize_answer_text(gold_answer_text(gold))
    return pred_text == gold_text, "text", "ok" if pred_text == gold_text else f"text pred={pred_text!r} gold={gold_text!r}"


def write_mismatches(path: Path, rows: list[EvalResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "line_no",
                "sample_id",
                "gold_id",
                "mode",
                "reason",
                "pred_answer",
                "gold_answer",
                "question",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "line_no": row.line_no,
                    "sample_id": row.sample_id,
                    "gold_id": row.gold_id,
                    "mode": row.mode,
                    "reason": row.reason,
                    "pred_answer": row.pred_answer,
                    "gold_answer": row.gold_answer,
                    "question": row.question,
                }
            )


def run(args: argparse.Namespace) -> int:
    gold_rows, by_id = load_gold(args.gold)
    total = 0
    matched = 0
    correct = 0
    valid_json = 0
    numeric_total = 0
    numeric_correct = 0
    text_total = 0
    text_correct = 0
    confidence_values: list[float] = []
    route_counts: collections.Counter[str] = collections.Counter()
    arbitration_counts: collections.Counter[str] = collections.Counter()
    match_source_counts: collections.Counter[str] = collections.Counter()
    bad_rows: list[EvalResult] = []

    for line_no, row in iter_jsonl(args.pred):
        total += 1
        if row.get("valid_json", True):
            valid_json += 1
        route_counts[str(row.get("route", "unknown"))] += 1
        arbitration_counts[str(row.get("arbitration", "unknown"))] += 1
        response = extract_response(row)
        confidence = response.get("confidence")
        if isinstance(confidence, (int, float)):
            confidence_values.append(float(confidence))
        gold, source = resolve_gold(row, line_no, gold_rows, by_id)
        match_source_counts[source] += 1
        if gold is None:
            bad_rows.append(
                EvalResult(
                    line_no=line_no,
                    sample_id=str(row.get("sample_id", "")),
                    gold_id="",
                    correct=False,
                    mode="unmatched",
                    pred_answer=str(response.get("answer", "")),
                    gold_answer="",
                    question=str(row.get("question", "")),
                    reason="could_not_match_gold",
                )
            )
            continue
        matched += 1
        pred_answer = str(response.get("answer", ""))
        ok, mode, reason = compare_answers(pred_answer, gold, args.rel_tol, args.abs_tol)
        if mode == "numeric":
            numeric_total += 1
            numeric_correct += int(ok)
        else:
            text_total += 1
            text_correct += int(ok)
        correct += int(ok)
        if not ok:
            bad_rows.append(
                EvalResult(
                    line_no=line_no,
                    sample_id=str(row.get("sample_id", "")),
                    gold_id=gold.sample_id,
                    correct=False,
                    mode=mode,
                    pred_answer=pred_answer,
                    gold_answer=gold_answer_text(gold),
                    question=gold.question,
                    reason=reason,
                )
            )

    print(f"file: {args.pred}")
    print(f"gold: {args.gold}")
    print(f"rows: {total}")
    print(f"matched_gold: {matched}/{total}")
    print(f"valid_json: {valid_json}/{total}")
    print(f"correct: {correct}/{matched}")
    print(f"accuracy: {correct / matched:.4f}" if matched else "accuracy: NA")
    if numeric_total:
        print(f"numeric_accuracy: {numeric_correct}/{numeric_total} = {numeric_correct / numeric_total:.4f}")
    if text_total:
        print(f"text_accuracy: {text_correct}/{text_total} = {text_correct / text_total:.4f}")
    if confidence_values:
        print(f"avg_confidence: {sum(confidence_values) / len(confidence_values):.4f}")
    print("match_source_counts:")
    for key, value in match_source_counts.most_common():
        print(f"  {key}: {value}")
    print("arbitration_counts:")
    for key, value in arbitration_counts.most_common():
        print(f"  {key}: {value}")
    print("route_counts:")
    for key, value in route_counts.most_common():
        print(f"  {key}: {value}")
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
    elif args.mismatches_out:
        write_mismatches(args.mismatches_out, [])
        print(f"mismatches_csv: {args.mismatches_out}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, required=True, help="JSONL output from python -m model.cli")
    parser.add_argument("--gold", type=Path, default=Path("data/Physics_Problems.csv"))
    parser.add_argument("--rel-tol", type=float, default=0.03, help="Relative tolerance for numeric answers")
    parser.add_argument("--abs-tol", type=float, default=1e-30, help="Absolute tolerance for numeric answers in SI units")
    parser.add_argument("--show", type=int, default=15)
    parser.add_argument("--mismatches-out", type=Path, default=Path("outputs/type2_eval_mismatches.csv"))
    return parser.parse_args()


if __name__ == "__main__":
    configure_stdio()
    raise SystemExit(run(parse_args()))
