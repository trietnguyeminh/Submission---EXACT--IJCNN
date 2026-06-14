from __future__ import annotations
import re

# Prefer explicit final-answer lines. Use the LAST one if the model repeats.
FIN = re.compile(r"Final Answer\s*[:\-]?\s*\(?\s*([A-D]|Yes|No|Unknown|Uncertain)\b", re.I)
CANNOT = re.compile(
    r"(cannot conclude|cannot determine|cannot be determined|"
    r"not enough information|no premise (?:states|provides|gives)|"
    r"insufficient information)",
    re.I,
)
SUPP = re.compile(r"Supporting Premises\s*[:\-]?\s*\[([^\]]*)\]", re.I)
PREM_MENTION = re.compile(r"[Pp]remises?\s+(\d{1,2})")


def _norm_label(v: str | None) -> str | None:
    if not v:
        return None
    s = str(v).strip()
    up = s.upper()
    if up in {"A", "B", "C", "D"}:
        return up
    title = s.title()
    if title in {"Yes", "No", "Unknown", "Uncertain"}:
        return title
    return None


def parse_final(raw: str) -> str | None:
    matches = FIN.findall(raw or "")
    if matches:
        return _norm_label(matches[-1])
    # Conservative fallback only when the model clearly says insufficiency.
    if CANNOT.search(raw or ""):
        return "Unknown"
    return None


def parse_premises_used(raw: str, n_premises: int) -> list[int]:
    """Convert 1-based citations in model output to 0-based indices, clipped."""
    ids: set[int] = set()
    m = SUPP.search(raw or "")
    if m:
        ids |= {int(x) for x in re.findall(r"\d+", m.group(1))}
    if not ids:
        ids |= {int(x) for x in PREM_MENTION.findall(raw or "")}
    return sorted(i - 1 for i in ids if 1 <= i <= n_premises)


def overlap_premises(question: str, premises: list[str], k: int = 2) -> list[int]:
    qw = set(re.findall(r"[a-z]{3,}", question.lower()))
    scored = sorted(
        ((len(qw & set(re.findall(r"[a-z]{3,}", p.lower()))), i) for i, p in enumerate(premises)),
        reverse=True,
    )
    return sorted(i for s, i in scored[:k] if s > 0)


def map_to_option(ans: str, options: list[str]) -> str:
    """Map a parsed label to EXACTLY one option string, as required by EXACT."""
    if not options:
        return ans
    ans = _norm_label(ans) or str(ans).strip()
    if ans in options:
        return ans
    low = {o.strip().lower(): o for o in options}
    a = ans.strip().lower()
    if a in low:
        return low[a]

    syn = {
        "unknown": ["uncertain", "cannot be determined", "undetermined", "not sure"],
        "uncertain": ["unknown", "cannot be determined"],
        "yes": ["true", "correct"],
        "no": ["false", "incorrect"],
    }
    for alt in syn.get(a, []):
        if alt in low:
            return low[alt]

    # Letter label -> exact letter option or option text starting with that letter.
    if len(a) == 1 and a.upper() in "ABCD":
        for o in options:
            if re.match(rf"^\(?{a}\)?[.):\s]", o.strip(), re.I) or o.strip().upper() == a.upper():
                return o
        idx = "ABCD".index(a.upper())
        if idx < len(options):
            return options[idx]

    # Text answer substring match.
    for o in options:
        if a and a in o.lower():
            return o

    # Safe schema fallback: for YNU options prefer Uncertain/Unknown, not options[0].
    for uncertain in ("uncertain", "unknown"):
        if uncertain in low:
            return low[uncertain]
    return options[0]
