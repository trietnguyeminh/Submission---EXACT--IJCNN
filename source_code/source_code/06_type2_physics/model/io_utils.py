from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_records(path: Path, question_column: str = "question", id_column: str | None = None) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or question_column not in reader.fieldnames:
                raise ValueError(f"CSV input must contain a {question_column!r} column")
            records = []
            for idx, row in enumerate(reader):
                sample_id = row.get(id_column) if id_column else None
                records.append({"id": sample_id or str(idx + 1), "question": row[question_column]})
            return records
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON input must be a list")
        records = []
        for idx, item in enumerate(data):
            if not isinstance(item, dict) or question_column not in item:
                raise ValueError(f"JSON item {idx} must contain {question_column!r}")
            records.append({"id": item.get(id_column or "id", str(idx + 1)), "question": item[question_column]})
        return records
    if suffix == ".jsonl":
        records = []
        with path.open(encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, dict) or question_column not in item:
                    raise ValueError(f"JSONL line {idx + 1} must contain {question_column!r}")
                records.append({"id": item.get(id_column or "id", str(idx + 1)), "question": item[question_column]})
        return records
    raise ValueError(f"Unsupported input format: {path}")


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

