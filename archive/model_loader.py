"""
model_loader.py -- Stage 0: Load Qwen2.5-7B voi quantization.

Supports 8-bit, 4-bit (bitsandbytes), hoac full-precision.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import PipelineConfig


class QwenModel:
    """Wrapper for Qwen2.5-7B-Instruct model + tokenizer."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self._loaded = False

    def load(self):
        """Load model va tokenizer voi quantization config."""
        cfg = self.config
        print(f"[Stage 0] Loading {cfg.model_id} ({cfg.quantization})...")
        print("  (Lan dau can tai ~15 GB tu HuggingFace, sau do cache lai)")

        # -- Quantization config
        bnb_cfg = self._make_bnb_config()

        # -- Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id,
            trust_remote_code=True,
            padding_side="left",
        )

        # -- Model
        load_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if bnb_cfg is not None:
            load_kwargs["quantization_config"] = bnb_cfg
        else:
            load_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, **load_kwargs
        )
        self.model.eval()
        self._loaded = True

        # -- Report VRAM usage
        if torch.cuda.is_available():
            used_gb = torch.cuda.memory_allocated() / 1024 ** 3
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            print(f"  Model loaded | VRAM: {used_gb:.1f} / {total_gb:.1f} GB")
        else:
            print("  Model loaded (CPU mode -- very slow, test only)")

    def generate(self, system: str, user: str, max_new_tokens: int = None) -> str:
        """Goi Qwen voi chat template, tra ve raw text output.

        Args:
            system: System prompt.
            user: User message.
            max_new_tokens: Override token limit (default: config.max_new_tokens).

        Returns:
            Raw generated text (stripped).
        """
        if not self._loaded:
            raise RuntimeError("Model chua duoc load. Goi .load() truoc.")

        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
                repetition_penalty=self.config.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_ids = output_ids[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    def _make_bnb_config(self):
        """Tao BitsAndBytesConfig dua tren config.quantization."""
        q = self.config.quantization
        if q == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        elif q == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        return None  # full precision


def print_system_info():
    """In thong tin he thong (PyTorch, CUDA, GPU)."""
    print(f"PyTorch  : {torch.__version__}")
    print(f"CUDA OK  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / 1024 ** 3
        print(f"GPU      : {props.name}  ({total_gb:.1f} GB)")
