"""
config.py -- Cau hinh trung tam cho Neuro-Symbolic Pipeline.

Chinh sua cac thong so tai day truoc khi chay pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    """Toan bo cau hinh cho pipeline."""

    # ── Model ──────────────────────────────────────────────────────
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    quantization: str = "8bit"  # '8bit' (~8-10 GB VRAM) | '4bit' (~5-6 GB) | 'none'

    # ── Dataset ────────────────────────────────────────────────────
    dataset_path: str = "Logic_Based_Educational_Queries-2.json"
    n_samples: int = 50  # so samples danh gia

    # ── Pipeline ───────────────────────────────────────────────────
    max_retries: int = 3  # so lan Qwen duoc phep sua lai khi Z3 loi
    max_new_tokens: int = 4096  # token sinh ra toi da cho formalization
    ans_max_tokens: int = 512  # token cho answer extraction

    # ── Generation ─────────────────────────────────────────────────
    temperature: float = 0.05
    repetition_penalty: float = 1.1
    do_sample: bool = True

    # ── Output ─────────────────────────────────────────────────────
    output_path: str = "pipeline_results_qwen.json"

    # ── Physics dataset (optional) ─────────────────────────────────
    physics_dataset_path: str = ""  # Duong dan toi file bai tap ly
    physics_n_samples: int = 50

    def __post_init__(self):
        """Validate config."""
        assert self.quantization in ("8bit", "4bit", "none"), (
            f"quantization phai la '8bit', '4bit', hoac 'none', nhan: {self.quantization!r}"
        )
        assert self.n_samples > 0, "n_samples phai > 0"
        assert self.max_retries >= 1, "max_retries phai >= 1"
        assert 0.0 <= self.temperature <= 2.0, "temperature phai trong [0, 2]"

    def summary(self) -> str:
        return (
            f"Config | Model: {self.model_id} | Quant: {self.quantization} | "
            f"N_SAMPLES={self.n_samples} | MAX_RETRIES={self.max_retries}"
        )
