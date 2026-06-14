# Modular Type 2 Physics Pipeline

This is the new structured pipeline for EXACT Type 2 physics inference. It keeps
the pipeline editable by separating retrieval, routing, solver use, LLM calls,
output validation, and CLI execution.

## Layout

- `config.py`: runtime configuration.
- `schemas.py`: dataclasses shared by the pipeline.
- `text_utils.py`: normalization, JSON extraction, answer cleanup.
- `io_utils.py`: CSV/JSON/JSONL loading and JSONL writing.
- `analyzer.py`: rule-based topic, target, and geometry classifier.
- `kb.py`: knowledge-base loader and metadata retrieval.
- `solver.py`: wrapper around the native deterministic physics solver.
- `solvers/formula_bank.py`: formula-bank and vector-geometry solver.
- `prompting.py`: system prompt and prompt builder.
- `llm_client.py`: OpenAI-compatible vLLM HTTP client.
- `ft_corrector.py`: optional local LoRA adapter corrector used by
  `--generator ft-corrector`.
- `validation.py`: JSON normalization and solver/LLM arbitration.
- `pipeline.py`: orchestration.
- `cli.py`: command-line entry point.
- `evaluate.py`: Type 2 evaluator.
- `requirements.txt`: Python dependencies for the modular pipeline.
- `../finetune/`: QLoRA dataset builder, training script, evaluation script,
  and 24GB GPU config for the physics corrector.

This package is self-contained. It does not import from any external pipeline
folder.

## Dry Run

```bash
python -m model.cli --dry-run --limit 1
```

## Solver-Only Test

```bash
python -m model.cli \
  --generator solver \
  --input data/Physics_Questions_Only.csv \
  --output outputs/type2_model_solver_test.jsonl \
  --limit 10
```

## Qwen/vLLM Inference

```bash
python -m model.cli \
  --generator qwen \
  --input data/Physics_Questions_Only.csv \
  --output outputs/type2_model_qwen_test.jsonl \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3-8B \
  --use-response-format \
  --limit 10
```

## Evaluate

```bash
python -m model.evaluate \
  --pred outputs/type2_model_qwen_test.jsonl \
  --gold data/Physics_Problems.csv
```

## Fine-tuned Corrector Inference

After training the LoRA adapter on a Linux GPU server:

```bash
python -m model.cli \
  --generator ft-corrector \
  --model Qwen/Qwen3-8B \
  --adapter-path outputs/qwen3_8b_exact_corrector_lora \
  --input data/Physics_Problems.csv \
  --output outputs/test_ft_corrector.jsonl \
  --question-column question \
  --id-column id
```

## Current Pipeline Flow

```text
question
 -> QueryAnalyzer
 -> KnowledgeBase.retrieve(formula_cards, geometry_cards)
 -> DeterministicPhysicsSolver
 -> PromptBuilder
 -> Qwen/vLLM or solver-only response
 -> output validation
 -> solver arbitration
 -> JSONL audit record
```

## Linux Server Setup

```bash
cd /path/to/code
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r model/requirements.txt
```
