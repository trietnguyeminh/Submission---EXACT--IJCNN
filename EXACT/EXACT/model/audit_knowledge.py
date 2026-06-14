from __future__ import annotations

import argparse
import collections
import csv
import json
import re
from pathlib import Path
from typing import Any

from .analyzer import QueryAnalyzer
from .config import PipelineConfig
from .kb import KnowledgeBase
from .physics_ir import build_physics_ir
from .solver import DeterministicPhysicsSolver
from .text_utils import configure_stdio


def _prefix(sample_id: str | None) -> str:
    match = re.match(r"[A-Za-z]+", sample_id or "")
    return match.group(0) if match else "NOID"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.00%"
    return f"{100.0 * numerator / denominator:.2f}%"


def build_audit(input_path: Path, kb_root: Path, question_column: str, id_column: str | None) -> dict[str, Any]:
    rows = _read_rows(input_path)
    analyzer = QueryAnalyzer()
    solver = DeterministicPhysicsSolver()
    kb = KnowledgeBase(kb_root)

    prefix_counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    domain_counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    law_counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    route_counts: collections.Counter[str] = collections.Counter()
    formula_card_counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    geometry_card_counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    example_card_counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    missing_examples: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)

    for row_number, row in enumerate(rows, start=1):
        question = str(row.get(question_column, ""))
        sample_id = str(row.get(id_column, "")) if id_column else str(row.get("id", row_number))
        analysis = analyzer.analyze(question)
        ir = build_physics_ir(question, analysis)
        result = solver.solve(question)
        context = kb.retrieve(analysis, formula_top_k=4, geometry_top_k=3, example_top_k=3)

        solved_key = "solved" if result.solved else "unsolved"
        prefix = _prefix(sample_id)
        prefix_counts[prefix][solved_key] += 1
        domain_counts[ir.domain][solved_key] += 1
        law_counts[f"{ir.domain}/{ir.law_family}"][solved_key] += 1
        route_counts[result.route] += 1

        for card in context.formula_cards:
            formula_card_counts[card.id][solved_key] += 1
        for card in context.geometry_cards:
            geometry_card_counts[card.id][solved_key] += 1
        for card in context.example_cards:
            example_card_counts[card.id][solved_key] += 1

        if not result.solved and len(missing_examples[f"{ir.domain}/{ir.law_family}"]) < 5:
            missing_examples[f"{ir.domain}/{ir.law_family}"].append(
                {
                    "row": row_number,
                    "id": sample_id,
                    "target": ir.target,
                    "knowns": ir.knowns,
                    "state_condition": ir.state_condition,
                    "geometry": ir.geometry,
                    "route": result.route,
                    "question": question[:240],
                }
            )

    return {
        "rows": len(rows),
        "by_prefix": _counter_table(prefix_counts),
        "by_domain": _counter_table(domain_counts),
        "by_law_family": _counter_table(law_counts),
        "by_route": route_counts.most_common(),
        "formula_card_coverage": _counter_table(formula_card_counts),
        "geometry_card_coverage": _counter_table(geometry_card_counts),
        "example_card_coverage": _counter_table(example_card_counts),
        "missing_examples": missing_examples,
    }


def _counter_table(counters: dict[str, collections.Counter[str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, counter in counters.items():
        solved = counter.get("solved", 0)
        unsolved = counter.get("unsolved", 0)
        total = solved + unsolved
        rows.append(
            {
                "name": name,
                "total": total,
                "solved": solved,
                "unsolved": unsolved,
                "solved_rate": solved / total if total else 0.0,
            }
        )
    rows.sort(key=lambda item: (-item["unsolved"], item["name"]))
    return rows


def print_audit(audit: dict[str, Any], *, top: int) -> None:
    print(f"rows: {audit['rows']}")
    for title, key in (
        ("By Prefix", "by_prefix"),
        ("By Domain", "by_domain"),
        ("By Law Family", "by_law_family"),
    ):
        print(f"\n{title}")
        for row in audit[key][:top]:
            print(
                f"{row['name']}: total={row['total']} solved={row['solved']} "
                f"unsolved={row['unsolved']} solved_rate={_pct(row['solved'], row['total'])}"
            )

    print("\nRoutes")
    for route, count in audit["by_route"][:top]:
        print(f"{route}: {count}")

    print("\nTop Missing Examples")
    for law, examples in list(audit["missing_examples"].items())[:top]:
        print(f"\n## {law}")
        for example in examples[:3]:
            print(
                f"- row={example['row']} id={example['id']} target={example['target']} "
                f"knowns={','.join(example['knowns']) or '-'} route={example['route']}"
            )
            print(f"  {example['question']}")


def main() -> int:
    configure_stdio()
    parser = argparse.ArgumentParser(description="Audit physics knowledge coverage by law family.")
    parser.add_argument("--input", type=Path, default=Path("data/Physics_Problems.csv"))
    parser.add_argument("--kb-root", type=Path, default=PipelineConfig.kb_root)
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--json", dest="json_output", type=Path, default=None)
    args = parser.parse_args()

    audit = build_audit(args.input, args.kb_root, args.question_column, args.id_column)
    print_audit(audit, top=args.top)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote JSON audit to {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
