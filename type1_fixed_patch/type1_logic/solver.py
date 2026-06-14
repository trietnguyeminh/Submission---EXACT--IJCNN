from __future__ import annotations
from app import config
from app.schemas import PredictRequest
from common.llm import chat
from . import parser as P
from .prompt import build_prompt
from .verifier_v35 import verify


def solve_type1(req: PredictRequest) -> dict:
    prompt = build_prompt(req.premises, req.query, req.options)
    raw = ""
    try:
        model = config.TYPE1_LORA or config.MODEL_ID
        raw = chat(
            config.VLLM_BASE_URL,
            model,
            "",
            prompt,
            timeout=config.T1_LLM_TIMEOUT,
            max_tokens=380,
        )
    except Exception as exc:
        raw = f"[llm_error] {exc}"

    ans = P.parse_final(raw) or "Unknown"
    used = P.parse_premises_used(raw, len(req.premises))
    reasoning = (
        {"type": "cot", "steps": [s.strip() for s in raw.split("\n") if s.strip()][:8]}
        if raw and not raw.startswith("[llm_error]")
        else None
    )

    # v35 symbolic verifier: proof-certified correction only. It abstains on MC/statement.
    if ans in ("Yes", "No", "Unknown", "Uncertain"):
        v_ans, v_prem, v_reason = verify(req.query, req.premises, ans)
        if v_ans is not None:
            ans = v_ans
            used = sorted(set(used) | set(v_prem)) or v_prem
            reasoning = {"type": "proof", "steps": [v_reason]}

    if not used:
        used = P.overlap_premises(req.query, req.premises)

    final = P.map_to_option(ans, req.options)
    expl = (
        raw.split("Final Answer")[0].strip()[:900]
        if raw and not raw.startswith("[llm_error]")
        else "Derived from the given premises by the logic pipeline."
    )
    return {
        "query_id": req.query_id,
        "answer": final,
        "unit": "",
        "explanation": expl or "Derived from the given premises.",
        "premises_used": used,
        "reasoning": reasoning,
    }
