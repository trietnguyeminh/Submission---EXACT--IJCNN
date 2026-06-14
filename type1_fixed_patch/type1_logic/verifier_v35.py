"""Conservative v35-style symbolic verifier for EXACT Type-1 NL premises.

This is NOT a full FOL/Z3 solver. It is a production-safe verifier that only
applies small proof-certified corrections. It abstains on statement-form and
unsupported patterns.

Rules:
  E1: forall not Q derivable => existential question over Q -> No
  PE: forall Q derivable OR exists Q given/derived => existential question over Q -> Yes
  U1: forall not Q derivable and no positive universal proof => universal question -> No
  PY: forall Q derivable and no negative universal proof => universal question -> Yes
"""
from __future__ import annotations
import re

IF_RE   = re.compile(r"^if\s+(?:a|an|the)?\s*(.+?),\s*then\s+(?:the|a|an)?\s*(.+?)\.?$", re.I)
EVERY   = re.compile(r"^(?:every|all|each)\s+(.+?)\.?$", re.I)
NOONE   = re.compile(r"^no\s+(.+?)\.?$", re.I)
ATLEAST = re.compile(r"^(?:at least one|some|there (?:is|exists))\s+(.+?)\.?$", re.I)
NEG     = re.compile(r"\b(?:does not|do not|doesn't|don't|never|cannot|can't|fails? to|not)\s+(.+)$", re.I)


DOMAIN_WORDS = {"student","students","intern","interns","researcher","researchers","course","courses","lab","member","members"}

def _strip_subject(rest: str) -> str:
    toks = rest.strip().split()
    while toks and re.sub(r"[^a-zA-Z]", "", toks[0]).lower() in DOMAIN_WORDS:
        toks = toks[1:]
    # remove relative connector after subject, e.g. "who reports"
    if toks and toks[0].lower() in {"who", "that"}:
        toks = toks[1:]
    return " ".join(toks)

STOP = {
    "the","a","an","their","his","her","its","all","every","each","some","there","exists",
    "does","do","did","is","are","can","will","would","should","could","based","above","premises",
    "following","statement","true","student","students","intern","interns","researcher","researchers",
    "course","courses","lab","member","members","person","people","who","that","least","one",
}


def _stem(w: str) -> str:
    w = w.lower()
    # light stemming for generated educational predicates: receives->receive, passes->pass, studies->study
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("es"):
        if w.endswith(("ses", "xes", "zes", "ches", "shes")):
            return w[:-2]
        return w[:-1]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    return w


def _norm(phrase: str) -> tuple[str, bool]:
    p = phrase.strip().lower().replace("?", " ")
    m = NEG.search(p)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip(" ."), True
    return re.sub(r"\s+", " ", p).strip(" ."), False


def _words(p: str) -> frozenset[str]:
    toks = []
    for raw in re.findall(r"[a-z]{2,}", p.lower()):
        w = _stem(raw)
        if len(w) >= 3 and w not in STOP:
            toks.append(w)
    return frozenset(toks)


def _match(a: frozenset[str], b: frozenset[str]) -> bool:
    if not a or not b:
        return False
    inter = len(a & b)
    # Exact/near containment for short predicate phrases.
    if inter >= min(len(a), len(b)) and inter >= 1:
        return True
    return inter / max(len(a), len(b)) >= 0.60


def parse_premises(premises: list[str]):
    edges, ufacts, nufacts, efacts = [], [], [], []
    for i, prem in enumerate(premises):
        s = prem.strip()
        m = IF_RE.match(s)
        if m:
            (a, na), (c, nc) = _norm(m.group(1)), _norm(m.group(2))
            A, C = _words(a), _words(c)
            if A and C:
                edges.append((A, na, C, nc, i))
                # Classical contrapositive.
                edges.append((C, not nc, A, not na, i))
            continue
        m = NOONE.match(s)
        if m:
            w = _words(_norm(_strip_subject(m.group(1)))[0])
            if w:
                nufacts.append((w, i)); continue
        m = EVERY.match(s)
        if m:
            (c, neg) = _norm(_strip_subject(m.group(1))); w = _words(c)
            if w:
                (nufacts if neg else ufacts).append((w, i)); continue
        m = ATLEAST.match(s)
        if m:
            w = _words(_norm(_strip_subject(m.group(1)))[0])
            if w:
                efacts.append((w, i))
    return edges, ufacts, nufacts, efacts


def _derive_universal(target: frozenset[str], want_neg: bool, edges, ufacts, nufacts):
    derived = {(w, False, (i,)) for w, i in ufacts} | {(w, True, (i,)) for w, i in nufacts}
    changed = True
    while changed:
        changed = False
        cur = list(derived)
        for (A, na, C, nc, ei) in edges:
            for (w, neg, path) in cur:
                if neg == na and _match(A, w):
                    key = (C, nc, path + (ei,))
                    if not any(c2 == C and n2 == nc for c2, n2, _ in derived):
                        derived.add(key); changed = True
    for (w, neg, path) in derived:
        if neg == want_neg and _match(w, target):
            return sorted(set(p for p in path if p >= 0))
    return None


def _derive_existential(target: frozenset[str], edges, efacts, ufacts):
    # Direct existential witness.
    frontier = [(w, (i,)) for w, i in efacts]
    seen = {(w, False) for w, _ in frontier}
    # Universal fact also counts as a positive existence witness in this dataset convention.
    for w, i in ufacts:
        frontier.append((w, (i,)))
        seen.add((w, False))
    for w, path in frontier:
        if _match(w, target):
            return sorted(set(path))
    changed = True
    while changed:
        changed = False
        cur = list(frontier)
        for (A, na, C, nc, ei) in edges:
            # Positive existential forward chaining only. Do not infer existential from negative edges here.
            if na or nc:
                continue
            for (w, path) in cur:
                if _match(A, w) and (C, False) not in seen:
                    seen.add((C, False))
                    new = (C, path + (ei,))
                    frontier.append(new)
                    changed = True
                    if _match(C, target):
                        return sorted(set(new[1]))
    return None


def _question_target(question: str) -> frozenset[str]:
    q = re.sub(r"^(do|does|is|are|can|will|would|should|could)\b", "", question.lower()).strip()
    q = re.sub(r"\bbased on the above premises\b", "", q)
    q = re.sub(r"\bwhich conclusion logically follows\b", "", q)
    return _words(_norm(q)[0])


def verify(question: str, premises: list[str], model_answer: str):
    """Return (verdict_answer|None, proof_premises(0-based), reason)."""
    edges, ufacts, nufacts, efacts = parse_premises(premises)
    if not (edges or ufacts or nufacts or efacts):
        return None, [], "no parseable premises"
    ql = question.lower()
    # Statement-form and MC are out of scope for production correction.
    if re.search(r"\bif\b.*\bthen\b|statement\s*:", ql):
        return None, [], "statement/conditional out of scope"
    if re.search(r"\bwhich\b|\boption\b|\bA\.", question):
        return None, [], "MC out of scope"

    is_exist = bool(re.search(r"\bat least one\b|\bsome\b|\bany\b|\bthere (?:is|exists)\b", ql))
    is_univ  = bool(re.search(r"\ball\b|\bevery\b|\beach\b", ql)) and not is_exist
    tgt = _question_target(question)
    if not tgt:
        return None, [], "no target"

    neg_path = _derive_universal(tgt, True, edges, ufacts, nufacts)
    pos_univ_path = _derive_universal(tgt, False, edges, ufacts, nufacts)
    pos_exist_path = _derive_existential(tgt, edges, efacts, ufacts)

    if is_exist:
        if neg_path is not None:
            return "No", neg_path, "E1: forall-not target derivable -> no instance exists"
        if pos_exist_path is not None or pos_univ_path is not None:
            return "Yes", (pos_exist_path or pos_univ_path), "PE: positive witness/universal proof establishes existence"
    if is_univ:
        if pos_univ_path is not None and neg_path is None:
            return "Yes", pos_univ_path, "PY: universal chain proves target"
        if neg_path is not None and pos_univ_path is None:
            return "No", neg_path, "U1: forall-not target derivable"
    return None, [], "abstain"
