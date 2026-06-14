#!/usr/bin/env python3
"""
main.py -- Neuro-Symbolic Pipeline: Qwen2.5-7B + Z3 (Local, No Cloud API)

Entry point cho EXACT 2026 (XAI Challenge @ IJCNN).

Usage:
  python main.py                                          # Chay voi default config
  python main.py --dataset path/to/dataset.json           # Chi dinh dataset
  python main.py --n-samples 10 --max-retries 5           # Override params
  python main.py --quantization 4bit                      # Dung 4-bit (~5 GB VRAM)
  python main.py --physics path/to/physics.json           # Them physics dataset
  python main.py --no-model                               # Dry-run (khong load model)

Pipeline 5 giai doan:
  Stage 0: Cai dat & Load Qwen2.5-7B (8-bit quantization)
  Stage 1: Data Grounding + Dual-Layer Ontology
  Stage 2: Local Ontology Generation + AST FOL (Qwen)
  Stage 3: Deterministic Z3 Compilation & Verification
  Stage 4: Feedback Loop (Z3 -> Qwen) + Answer Extraction
  Stage 5: Evaluation & Export
"""

import argparse
import json
import sys
import time

from config import PipelineConfig
from model_loader import QwenModel, print_system_info
from pipeline import PipelineResult, run_pipeline
from evaluation import (
    evaluate,
    print_summary,
    print_per_sample,
    export_results,
)
from ontology import GLOBAL_ONTOLOGY


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Neuro-Symbolic Pipeline: Qwen2.5-7B + Z3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model ID",
    )
    p.add_argument(
        "--quantization",
        choices=["8bit", "4bit", "none"],
        default="8bit",
        help="Quantization mode",
    )
    p.add_argument(
        "--dataset",
        default="Logic_Based_Educational_Queries-2.json",
        help="Path to logic dataset JSON",
    )
    p.add_argument(
        "--physics",
        default="",
        help="Path to physics dataset JSON (optional)",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of samples to evaluate",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max Z3 feedback retries per sample",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Max tokens for formalization generation",
    )
    p.add_argument(
        "--output",
        default="pipeline_results_qwen.json",
        help="Output JSON path",
    )
    p.add_argument(
        "--no-model",
        action="store_true",
        help="Dry-run: skip model loading (test pipeline logic only)",
    )
    return p.parse_args()


def load_dataset(path: str, n_samples: int) -> list:
    """Load va slice dataset.

    Supports two formats:
      - Logic dataset: list of { premises-NL, questions, answers, ... }
      - Physics dataset: list of { question, answer, ... } (will be normalized)
    """
    with open(path, encoding="utf-8") as f:
        full_dataset = json.load(f)

    samples = full_dataset[:n_samples]
    print(f"\nDataset: {path}")
    print(f"  Total: {len(full_dataset)} samples -> dung {len(samples)}")

    if samples:
        keys = list(samples[0].keys())
        print(f"  Fields: {keys}")

        # Stats cho logic dataset
        if "premises-NL" in samples[0]:
            q_counts = [len(s.get("questions", [])) for s in samples]
            p_counts = [len(s.get("premises-NL", [])) for s in samples]
            print(f"  Avg premises/sample : {sum(p_counts) / len(p_counts):.1f}")
            print(f"  Avg questions/sample: {sum(q_counts) / len(q_counts):.1f}")
        # Stats cho physics dataset
        elif "question" in samples[0]:
            print("  [Physics format detected -- normalizing fields]")
            # Normalize physics format to match pipeline expectations
            for s in samples:
                if "premises-NL" not in s and "premises" not in s:
                    # Physics problems don't have premises; use question context
                    s["premises-NL"] = [s.get("question", "")]
                if "questions" not in s:
                    s["questions"] = [s.get("question", "")]
                if "answers" not in s:
                    ans = s.get("answer", s.get("final_answer", "Unknown"))
                    s["answers"] = [str(ans)]

    return samples


def run_all(
    samples: list,
    qwen: QwenModel,
    config: PipelineConfig,
    is_physics: bool = False,
) -> list:
    """Chay pipeline cho toan bo samples voi checkpoint reporting."""
    all_results = []

    tag = "Physics" if is_physics else "Logic"
    print("=" * 65)
    print(f"  Neuro-Symbolic Pipeline -- Qwen2.5-7B + Z3 [{tag}]")
    print(f"  Model   : {config.model_id}  ({config.quantization})")
    print(f"  Samples : {len(samples)}   Max retries: {config.max_retries}")
    print("=" * 65)

    for idx, sample in enumerate(samples):
        try:
            r = run_pipeline(idx, sample, qwen, config, is_physics)
            all_results.append(r)
        except Exception as fatal:
            print(f"  [Sample {idx:02d}] FATAL: {fatal}")
            r = PipelineResult(
                sample_id=idx,
                status="failed",
                ground_truth=sample.get("answers", []),
                total_questions=len(sample.get("questions", [])),
                error_log=[str(fatal)],
            )
            all_results.append(r)

        # Checkpoint every 10 samples
        if (idx + 1) % 10 == 0:
            done = idx + 1
            correct = sum(r.correct_count for r in all_results)
            total_q = sum(r.total_questions for r in all_results)
            acc = correct / total_q if total_q else 0
            sat_cnt = sum(1 for r in all_results if r.z3_status == "sat")
            print(
                f"\n  -- Checkpoint {done}/{len(samples)} | "
                f"Acc: {acc:.1%} | Z3-sat: {sat_cnt}/{done} --\n"
            )

        time.sleep(0.1)

    print(f'\n{"=" * 65}')
    print(f"  Pipeline xong -- {len(all_results)} samples [{tag}]")
    print(f'{"=" * 65}')

    return all_results


def main():
    """Main entry point."""
    args = parse_args()

    # ── Build config ──────────────────────────────────────────────
    config = PipelineConfig(
        model_id=args.model_id,
        quantization=args.quantization,
        dataset_path=args.dataset,
        n_samples=args.n_samples,
        max_retries=args.max_retries,
        max_new_tokens=args.max_new_tokens,
        output_path=args.output,
        physics_dataset_path=args.physics,
    )

    # ── Stage 0: System info & Model loading ──────────────────────
    print_system_info()
    print(config.summary())

    # Global Ontology info
    print("\nGlobal Ontology loaded:")
    for k, v in GLOBAL_ONTOLOGY.items():
        print(f"  {k}: {v}")

    # Load model (unless dry-run)
    qwen = QwenModel(config)
    if not args.no_model:
        qwen.load()
    else:
        print("\n[DRY-RUN] Model loading skipped (--no-model)")

    # ── Stage 1: Load Dataset ─────────────────────────────────────
    samples = load_dataset(config.dataset_path, config.n_samples)

    if not samples:
        print("ERROR: No samples loaded. Check dataset path.")
        sys.exit(1)

    # ── Stages 2-4: Run pipeline ──────────────────────────────────
    all_results = run_all(samples, qwen, config, is_physics=False)

    # ── Stage 5: Evaluate & Export ────────────────────────────────
    metrics = evaluate(all_results)
    print_summary(metrics, config.model_id, config.quantization)
    print_per_sample(all_results)
    export_results(all_results, metrics, config, config.output_path)

    # ── Physics dataset (optional) ────────────────────────────────
    if config.physics_dataset_path:
        print("\n\n" + "=" * 65)
        print("  PHYSICS DATASET")
        print("=" * 65)

        physics_samples = load_dataset(
            config.physics_dataset_path,
            config.physics_n_samples,
        )
        if physics_samples:
            physics_results = run_all(
                physics_samples, qwen, config, is_physics=True
            )
            physics_metrics = evaluate(physics_results)
            print_summary(physics_metrics, config.model_id, config.quantization)
            print_per_sample(physics_results)

            physics_output = config.output_path.replace(".json", "_physics.json")
            export_results(
                physics_results, physics_metrics, config, physics_output
            )

    print("\n Pipeline hoan tat!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
