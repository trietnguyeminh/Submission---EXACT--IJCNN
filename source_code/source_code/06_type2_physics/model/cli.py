from __future__ import annotations

import argparse
from pathlib import Path

from .config import PipelineConfig
from .pipeline import PhysicsPipeline
from .text_utils import configure_stdio


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the modular EXACT Type 2 physics pipeline.")
    parser.add_argument("--input", dest="input_path", type=Path, default=Path("data/Physics_Questions_Only.csv"))
    parser.add_argument("--output", dest="output_path", type=Path, default=Path("outputs/type2_model_outputs.jsonl"))
    parser.add_argument("--kb-root", type=Path, default=Path("knowledge_base/physics"))
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--generator", choices=["qwen", "solver", "ft-corrector"], default="qwen")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-model-check", action="store_true")
    parser.add_argument("--use-response-format", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--prefer-solver", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--ft-base-model", default=None)
    parser.add_argument("--ft-max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--formula-top-k", type=int, default=4)
    parser.add_argument("--geometry-top-k", type=int, default=3)
    parser.add_argument("--example-top-k", type=int, default=3)
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(**vars(args))


def main() -> int:
    configure_stdio()
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        pipeline = PhysicsPipeline(config_from_args(args))
        return pipeline.run_batch()
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
