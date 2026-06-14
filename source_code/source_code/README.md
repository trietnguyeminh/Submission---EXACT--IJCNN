# EXACT 2026 — Neuro-Symbolic QA Pipeline (Source Code)

A neuro-symbolic system: an LLM solves the problem, a LoRA adapter teaches it the
logic-QA format, a strict parser extracts the answer, a symbolic proof engine verifies
and corrects certain logic errors, and a safety/selector layer prevents unjustified edits.
Type 2 (physics) is solved by a deterministic formula pipeline with an LLM fallback.

Served live via vLLM (OpenAI-compatible) for committee verification — see 00_deployment.

## Layer map

| Layer | Folder / file | Role |
|---|---|---|
| 0. Deployment (live API) | `00_deployment/exact_vllm_kaggle_submission.ipynb` | vLLM OpenAI server (Qwen3-8B + LoRA, TP=2 on 2×T4) + FastAPI `/predict`, proxies `/v1/models` & `/v1/completions`, Cloudflare tunnel. Type1 → LoRA+verifier, Type2 → physics pipeline. |
| 1. Training (LoRA) | `01_training_lora/notebook_v14_cot_lora_finetune.ipynb` | Fine-tune Qwen3-8B with LoRA on the logic dataset. Target = `explanation` (chain-of-thought) + `Final Answer`. Output = LoRA adapter (not a new full model). |
| 2. Inference | `02_inference/notebook_full_model_eval_v2_flatten.ipynb`, `00_rebuild_from_scratch_two_tests_v35.ipynb` | Flatten records→questions, build the LOCKED prompt ("Use only the given premises…", "End with: Final Answer: X"), Qwen+LoRA generate raw reasoning. |
| 3. Type1 symbolic | `03_type1_symbolic/type1_logic/` | `parser.py` (strict Final-Answer extraction, no stray Yes/No), parsefix (format repair), `verifier_v35.py` (proof engine over premises-FOL; rule E1: ∀x¬Q(x) ⇒ ¬∃x Q(x); flips to a certain answer, else keeps the model), `solver.py`, `prompt.py`, `tests/`. |
| 4. Evaluation / safety | `04_evaluation_safety/` | Reload-verify (post-v30 artifact safety: never trust a summary; recompute from file and assert match), metrics (acc, macro/weighted-F1, per-label, MC/YNU split, confusion, error cases), subset benchmark tagging, master selector, v35 approval layer. |
| 5. Research (analysis-only) | `05_research_v36_v37/notebook_05_prompt_locked_v35_fullrun_v36_analysis.ipynb` | v36 (MC option verifier) & v37 (statement verifier): analysis-only, report candidates and precision; applied to output ONLY if precision ≥ 0.9 on both test sets. Currently NOT applied. |
| 6. Type2 physics | `06_type2_physics/model/`, `06_type2_physics/knowledge_base/` | Deterministic formula-bank solver + KB retrieval (analyzer→retrieve→solve→arbitrate), with Qwen fallback via vLLM. Returns numeric answer + ASCII unit. Non-LLM solver does not count toward the 8B limit. |

## One-line flow
Data → LoRA-finetuned Qwen → prompt-locked inference → strict parser → parsefix →
v35 proof verifier → safe selector → metrics/risk report → (v36/v37 analysis hooks).

## Compliance
- Single 8B-class LLM (Qwen3-8B; LoRA r=16 adapter). Only one LLM loaded/running at a time.
- All LLM inference served via vLLM OpenAI-compatible server; `/v1/models` verifiable.
- No third-party inference APIs. Symbolic verifier + physics solver are non-LLM tools.

Test sets: `generated_v4style_300` (regression: reproduce old v35) and
`benchmark_v2.1_1000` (balanced full benchmark).
