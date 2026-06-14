from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

from .schemas import KnowledgeCard, QueryAnalysis, RetrievedContext
from .text_utils import normalize_text


class KnowledgeBase:
    def __init__(self, root: Path):
        self.root = root
        self.sources = self._load_sources(root / "sources.csv")
        self.formula_cards = self._load_cards(root / "formula_cards.jsonl", kind="formula")
        self.geometry_cards = self._load_cards(root / "geometry_cards.jsonl", kind="geometry")
        self.example_cards = self._load_cards(root / "example_cards.jsonl", kind="example")

    def retrieve(
        self,
        analysis: QueryAnalysis,
        *,
        formula_top_k: int = 3,
        geometry_top_k: int = 2,
        example_top_k: int = 0,
    ) -> RetrievedContext:
        ranked_formulas = self._rank(self.formula_cards, analysis)
        topic_matched = [card for card in ranked_formulas if any(reason.startswith("topic:") for reason in card.score_reasons)]
        formula_cards = (topic_matched or ranked_formulas)[:formula_top_k]

        needs_geometry = bool(analysis.geometry_tags) or analysis.target_quantity in {"force", "electric_field"}
        geometry_cards = self._rank(self.geometry_cards, analysis)[:geometry_top_k] if needs_geometry else []
        example_cards = self._rank(self.example_cards, analysis)[:example_top_k] if example_top_k > 0 else []
        source_ids = sorted({card.source_id for card in formula_cards + geometry_cards + example_cards})
        return RetrievedContext(
            formula_cards=formula_cards,
            geometry_cards=geometry_cards,
            example_cards=example_cards,
            source_ids=source_ids,
        )

    def _rank(self, cards: list[KnowledgeCard], analysis: QueryAnalysis) -> list[KnowledgeCard]:
        ranked = []
        for card in cards:
            score, reasons = self._score(card.raw, analysis)
            if score <= 0:
                continue
            ranked.append(
                KnowledgeCard(
                    id=card.id,
                    source_id=card.source_id,
                    kind=card.kind,
                    raw=card.raw,
                    score=score,
                    score_reasons=reasons,
                )
            )
        ranked.sort(key=lambda card: card.score, reverse=True)
        return ranked

    def _score(self, raw: dict[str, Any], analysis: QueryAnalysis) -> tuple[float, list[str]]:
        question = analysis.normalized_question.lower()
        tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z_0-9]+", question))
        score = 0.0
        reasons: list[str] = []

        topic = str(raw.get("topic", "")).lower()
        if topic and self._topic_matches(topic, analysis.topic):
            score += 4.0
            reasons.append(f"topic:{topic}")

        targets = [str(item).lower() for item in raw.get("target_quantities", [])]
        if not targets and raw.get("target"):
            targets = [str(raw.get("target")).lower()]
        if analysis.target_quantity in targets:
            score += 4.0
            reasons.append(f"target:{analysis.target_quantity}")
        elif targets and any(self._target_matches(target, analysis.target_quantity) for target in targets):
            score += 2.0
            reasons.append("target_family")

        geometry_tags = [str(item).lower() for item in raw.get("geometry_tags", [])]
        for tag in analysis.geometry_tags:
            if tag.lower() in geometry_tags:
                score += 4.0
                reasons.append(f"geometry:{tag}")

        phrases = [str(item).lower() for item in raw.get("retrieval_phrases", [])]
        phrase_hits = [phrase for phrase in phrases if phrase and phrase in question]
        if phrase_hits:
            score += min(3.0, 1.5 * len(phrase_hits))
            reasons.append("phrase:" + ",".join(phrase_hits[:2]))

        text_blob = normalize_text(json.dumps(raw, ensure_ascii=False)).lower()
        card_tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z_0-9]+", text_blob))
        overlap = len(tokens & card_tokens)
        if overlap:
            score += min(2.0, math.log1p(overlap))
            reasons.append(f"token_overlap:{overlap}")

        if analysis.inverse_or_constraint and self._contains_any(text_blob, ["inverse", "zero", "constraint", "solve"]):
            score += 1.5
            reasons.append("inverse")

        if analysis.target_quantity in {"force", "electric_field"} and raw.get("kind") == "formula":
            if "vector_formula" in raw:
                score += 1.0
                reasons.append("vector_formula")

        return score, reasons

    @staticmethod
    def _topic_matches(card_topic: str, query_topic: str) -> bool:
        if card_topic == query_topic:
            return True
        if query_topic == "electrostatics" and card_topic == "electrostatics":
            return True
        if query_topic == "magnetism_induction" and card_topic in {"magnetism_induction", "inductor"}:
            return True
        if query_topic == "ac_lc_rlc" and card_topic in {"ac_lc_rlc", "inductor", "capacitor"}:
            return True
        return False

    @staticmethod
    def _target_matches(card_target: str, query_target: str) -> bool:
        families = {
            "force": {"net_force", "resultant_force", "force_magnitude"},
            "electric_field": {"field_strength", "field_intensity"},
            "frequency": {"angular_frequency", "period"},
            "emf": {"current_change", "time_interval"},
            "error": {"relative_error", "percent_error", "absolute_error"},
        }
        return card_target in families.get(query_target, set())

    @staticmethod
    def _contains_any(text: str, needles: list[str]) -> bool:
        return any(needle in text for needle in needles)

    @staticmethod
    def _load_sources(path: Path) -> dict[str, dict[str, str]]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8", newline="") as handle:
            return {row["source_id"]: row for row in csv.DictReader(handle)}

    @staticmethod
    def _load_cards(path: Path, *, kind: str) -> list[KnowledgeCard]:
        cards: list[KnowledgeCard] = []
        if not path.exists():
            return cards
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                raw.setdefault("kind", kind)
                cards.append(
                    KnowledgeCard(
                        id=str(raw.get("id") or f"{kind}_{line_no}"),
                        source_id=str(raw.get("source_id") or "unknown"),
                        kind=kind,
                        raw=raw,
                    )
                )
        return cards
