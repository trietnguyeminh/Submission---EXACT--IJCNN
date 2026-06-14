"""LOCKED Type-1 prompt.

This prompt mirrors `full_model_eval_v2_flatten_preds.json` from the current
Type-1 artifact zip. Do not hand-edit without re-running prompt-lock tests.
"""
from __future__ import annotations

PREFIX = (
    "You are solving a logic-based educational QA problem. "
    "Use only the given premises. Do not use outside knowledge.\n\n"
)

YNU_SUFFIX = (
    "\n\nReason step by step briefly, cite supporting premises if useful, "
    "and End with exactly one line: Final Answer: <Yes, No, or Unknown>\n"
)

MC_SUFFIX = (
    "\n\nReason step by step briefly, cite supporting premises if useful, "
    "and End with exactly one line: Final Answer: <A, B, C, or D>\n"
)

_YNU = {"yes", "no", "unknown", "uncertain", "true", "false"}


def _is_mc_options(options: list[str] | None) -> bool:
    if not options:
        return False
    cleaned = [str(o).strip().lower() for o in options]
    if all(o in _YNU for o in cleaned):
        return False
    # EXACT Type-1 MC may pass either ["A","B","C","D"] or full option strings.
    return len(options) >= 2


def build_prompt(premises: list[str], question: str, options: list[str] | None = None) -> str:
    prem = "\n".join(f"{i+1}. {p}" for i, p in enumerate(premises))
    suffix = MC_SUFFIX if _is_mc_options(options) else YNU_SUFFIX
    return f"{PREFIX}Premises:\n{prem}\n\nQuestion:\n{question.strip()}{suffix}"
