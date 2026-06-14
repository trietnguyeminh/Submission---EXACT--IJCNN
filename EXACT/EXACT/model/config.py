from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    input_path: Path = Path("data/Physics_Questions_Only.csv")
    output_path: Path = Path("outputs/type2_model_outputs.jsonl")
    kb_root: Path = Path("knowledge_base/physics")
    question_column: str = "question"
    id_column: str | None = None
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "Qwen/Qwen3-8B"
    generator: str = "qwen"
    start: int = 0
    limit: int | None = None
    append: bool = False
    dry_run: bool = False
    skip_model_check: bool = False
    use_response_format: bool = False
    enable_thinking: bool = False
    prefer_solver: bool = True
    adapter_path: Path | None = None
    ft_base_model: str | None = None
    ft_max_new_tokens: int = 512
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: float = 90.0
    sleep: float = 0.0
    formula_top_k: int = 4
    geometry_top_k: int = 3
    example_top_k: int = 3
