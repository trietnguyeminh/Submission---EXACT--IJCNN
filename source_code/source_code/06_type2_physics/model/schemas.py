from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class QueryAnalysis:
    question: str
    normalized_question: str
    topic: str
    target_quantity: str
    geometry_tags: list[str] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)
    inverse_or_constraint: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class KnowledgeCard:
    id: str
    source_id: str
    kind: str
    raw: dict[str, Any]
    score: float = 0.0
    score_reasons: list[str] = field(default_factory=list)

    def brief(self) -> dict[str, Any]:
        data = {
            "id": self.id,
            "source_id": self.source_id,
            "kind": self.kind,
            "score": round(self.score, 3),
            "score_reasons": self.score_reasons[:5],
        }
        for key in (
            "topic",
            "subtopic",
            "law_family",
            "target_quantities",
            "target",
            "geometry_tags",
            "input_shape",
            "formula",
            "vector_formula",
            "formula_steps",
            "conditions",
            "solver_hints",
            "anti_patterns",
            "answer_cardinality",
        ):
            if key in self.raw:
                data[key] = self.raw[key]
        return data


@dataclass
class RetrievedContext:
    formula_cards: list[KnowledgeCard] = field(default_factory=list)
    geometry_cards: list[KnowledgeCard] = field(default_factory=list)
    example_cards: list[KnowledgeCard] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "formula_cards": [card.brief() for card in self.formula_cards],
            "geometry_cards": [card.brief() for card in self.geometry_cards],
            "example_cards": [card.brief() for card in self.example_cards],
            "source_ids": self.source_ids,
        }


@dataclass
class SolverResult:
    solved: bool
    route: str
    answer: str
    fol: str
    cot: list[str]
    premises: list[str]
    confidence: float
    target: str
    value_si: float | None = None
    unit: str | None = None
    warning: str | None = None
    engine: str = "formula_bank"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineRecord:
    item_index: int
    sample_id: str | None
    question: str
    response: dict[str, Any]
    route: str
    solver_result: dict[str, Any] | None
    arbitration: str
    valid_json: bool
    validation_errors: list[str]
    raw_response: str | None
    latency_sec: float
    analysis: dict[str, Any]
    retrieval: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
