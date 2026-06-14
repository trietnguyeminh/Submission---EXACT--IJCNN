# Neuro-Symbolic Pipeline: Qwen2.5-7B + Z3

> **EXACT 2026** — The 2nd International XAI Challenge for Transparent Educational QA  
> IEEE IJCNN 2026 | URA Research Group (HCMUT)

Pipeline **5 giai đoạn** với Qwen2.5-7B chạy tại chỗ (8-bit, ~8-10 GB VRAM):

| Stage | Name | Technology |
|-------|------|-----------|
| **0** | Setup & Load Qwen | `transformers` + `bitsandbytes` |
| **1** | Data Grounding + Dual-Layer Ontology | Static JSON + Python |
| **2** | Local Ontology Generation + AST FOL | Qwen2.5-7B-Instruct (8-bit) |
| **3** | Deterministic Z3 Compilation & Verification | Z3Py (pure Python, no AI) |
| **4** | Feedback Loop (Z3 → Qwen) + Answer Extraction | Qwen2.5-7B-Instruct |
| **5** | Evaluation & Export | Python |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DUAL-LAYER ONTOLOGY                       │
│  ┌──────────────────┐  ┌──────────────────────────────────┐ │
│  │  Global Ontology  │  │     Local Ontology (per sample)  │ │
│  │  (immutable)      │  │     (Qwen-generated)             │ │
│  │  forall, exists,  │  │     Student(x), GPA_High(x), ... │ │
│  │  and, or, implies │  │                                  │ │
│  └──────────────────┘  └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Qwen2.5-7B Formalization                          │
│  Premises NL ──→ Local Ontology + AST JSON (recursive tree) │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: Z3 Deterministic Compiler                         │
│  AST JSON ──→ Z3 expressions ──→ satisfiability check       │
└────────────────────────┬────────────────────────────────────┘
                         │
                    ┌────┴────┐
                    │ Z3 OK?  │
                    └────┬────┘
                   yes   │   no
                    │    │    │
                    ▼    │    ▼
               ┌────┐   │  ┌──────────────────────┐
               │Done│   │  │ Stage 4a: Feedback    │
               └────┘   │  │ Error → Qwen → Retry  │
                         │  └──────────┬───────────┘
                         │             │
                         └─────────────┘ (max MAX_RETRIES)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 4b: Answer Extraction                                │
│  FOL context + Questions ──→ Qwen ──→ Predicted answers     │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Google Colab (Recommended)

Upload `colab_pipeline.py` to Colab and run cell-by-cell:

```python
# Upload dataset first
from google.colab import files
uploaded = files.upload()  # Select Logic_Based_Educational_Queries-2.json
```

Then run `colab_pipeline.py` (the all-in-one version).

### 2. Local / Modular Version

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests (no GPU needed)
python test_pipeline.py

# Run pipeline
python main.py --dataset Logic_Based_Educational_Queries-2.json --n-samples 50

# Run with 4-bit quantization (lower VRAM)
python main.py --quantization 4bit --n-samples 10

# Dry run (skip model loading)
python main.py --no-model --dataset sample_dataset.json --n-samples 3
```

### 3. API Server (for competition submission)

```bash
python api_server.py --port 8000

# Test endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "premises-NL": ["All students pass.", "John is a student."],
    "questions": ["Does John pass? (A) Yes (B) No"]
  }'
```

## Project Structure

```
neuro-symbolic-pipeline/
├── main.py              # Entry point (CLI)
├── config.py            # Configuration dataclass
├── model_loader.py      # Stage 0: Qwen model loading
├── ontology.py          # Stage 1: Dual-layer ontology + prompts
├── json_parser.py       # Robust JSON parser for LLM output
├── z3_compiler.py       # Stage 3: AST → Z3 compiler
├── pipeline.py          # Stages 2-4: Pipeline orchestrator
├── evaluation.py        # Stage 5: Metrics & export
├── api_server.py        # HTTP API for competition
├── colab_pipeline.py    # All-in-one for Google Colab
├── test_pipeline.py     # Unit tests (no GPU needed)
├── sample_dataset.json  # 3-sample test dataset
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Datasets

### Type 1: Logic-Based Educational Queries
- 464 records, 913 questions
- University regulations (grading, enrollment, scholarships)
- Premises in NL + FOL, multiple question types

### Type 2: Physics Problems
- 5,520 text-based physics problems
- Electric circuits & electrostatics
- Multi-step computation with CoT reasoning

## Key Bug Fixes

| # | Bug | Fix |
|---|-----|-----|
| **1** | `run_correction(None, None)` → `AttributeError` | None-guard + re-formalize on None |
| **2** | `rfind('}')` catches closing brace in explanation text | Brace-balancing parser |
| **3** | `a in var_map` → `TypeError: unhashable dict` | `_resolve_predicate_arg()` handler |
| **4** | `bound_variables` as dict instead of string | `_resolve_bound_var_name()` handler |
| **5** | Early return on retry failure skips answer extraction | `break` instead of `return` |

## Competition Requirements

- ✅ Open-source LLM ≤ 8B parameters (Qwen2.5-7B)
- ✅ Symbolic reasoning (Z3 Solver)
- ✅ API endpoint (`api_server.py`)
- ✅ Transparent reasoning (Local Ontology + AST + Z3 verification)
- ✅ Explanation quality (per-question reasoning output)
