#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install pyyaml before running train_qlora.py") from exc
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_max_memory(value: Any) -> dict[int | str, str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return {idx: str(item) for idx, item in enumerate(value)}
    if isinstance(value, dict):
        normalized: dict[int | str, str] = {}
        for key, item in value.items():
            if str(key).isdigit():
                normalized[int(key)] = str(item)
            else:
                normalized[str(key)] = str(item)
        return normalized
    raise ValueError("max_memory must be a list or mapping, e.g. ['20GiB', '20GiB']")


def render_messages(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    rendered = []
    for message in messages:
        rendered.append(f"{message['role'].title()}: {message['content']}")
    if add_generation_prompt:
        rendered.append("Assistant:")
    return "\n".join(rendered)


def build_dataset(tokenizer: Any, rows: list[dict[str, Any]], max_seq_length: int):
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets before running train_qlora.py") from exc

    def encode(row: dict[str, Any]) -> dict[str, list[int]]:
        messages = row["messages"]
        prompt = render_messages(tokenizer, messages[:-1], add_generation_prompt=True)
        full = render_messages(tokenizer, messages, add_generation_prompt=False)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full, truncation=True, max_length=max_seq_length, add_special_tokens=False)["input_ids"]
        labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[min(len(prompt_ids), len(full_ids)) :]
        labels = labels[: len(full_ids)]
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}

    return Dataset.from_list(rows).map(encode, remove_columns=list(rows[0].keys()))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a QLoRA physics corrector adapter.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    try:
        import torch
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install torch, transformers, peft, datasets, bitsandbytes, accelerate before training."
        ) from exc

    base_model = cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype_name = str(cfg.get("bnb_4bit_compute_dtype", "bfloat16")).lower()
    compute_dtype = torch.bfloat16 if dtype_name in {"bf16", "bfloat16"} else torch.float16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=bool(cfg.get("load_in_4bit", True)),
        bnb_4bit_quant_type=str(cfg.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bool(cfg.get("bnb_4bit_use_double_quant", True)),
    )
    model_kwargs = {
        "quantization_config": quant_config,
        "device_map": cfg.get("device_map", "auto"),
        "trust_remote_code": True,
    }
    max_memory = normalize_max_memory(cfg.get("max_memory"))
    if max_memory:
        model_kwargs["max_memory"] = max_memory
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)))

    lora_config = LoraConfig(
        r=int(cfg.get("lora_r", 32)),
        lora_alpha=int(cfg.get("lora_alpha", 64)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.get("target_modules", [])),
    )
    try:
        from peft import get_peft_model
    except ImportError as exc:
        raise RuntimeError("Installed peft package does not expose get_peft_model") from exc
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    max_seq_length = int(cfg.get("max_seq_length", 3072))
    train_rows = load_jsonl(Path(cfg["train_file"]))
    valid_rows = load_jsonl(Path(cfg["valid_file"]))
    train_dataset = build_dataset(tokenizer, train_rows, max_seq_length)
    valid_dataset = build_dataset(tokenizer, valid_rows, max_seq_length)

    output_dir = str(cfg["output_dir"])
    use_bf16 = bool(cfg.get("bf16", True))
    use_fp16 = bool(cfg.get("fp16", False))
    if use_bf16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        use_bf16 = False
        use_fp16 = True
    common_args = dict(
        output_dir=output_dir,
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 16)),
        num_train_epochs=float(cfg.get("num_train_epochs", 3)),
        learning_rate=float(cfg.get("learning_rate", 1.5e-4)),
        lr_scheduler_type=str(cfg.get("lr_scheduler_type", "cosine")),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        optim=str(cfg.get("optim", "paged_adamw_8bit")),
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        save_strategy=str(cfg.get("save_strategy", "steps")),
        save_steps=int(cfg.get("save_steps", 100)),
        logging_steps=int(cfg.get("logging_steps", 10)),
        save_total_limit=int(cfg.get("save_total_limit", 3)),
        per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", 1)),
        prediction_loss_only=bool(cfg.get("prediction_loss_only", True)),
        report_to=[],
    )
    try:
        training_args = TrainingArguments(
            **common_args,
            eval_strategy=str(cfg.get("eval_strategy", "steps")),
            eval_steps=int(cfg.get("eval_steps", 100)),
        )
    except TypeError:
        training_args = TrainingArguments(
            **common_args,
            evaluation_strategy=str(cfg.get("eval_strategy", "steps")),
            eval_steps=int(cfg.get("eval_steps", 100)),
        )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": valid_dataset,
        "data_collator": DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8),
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    resume_from_checkpoint = cfg.get("resume_from_checkpoint", None)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved LoRA adapter to {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - command-line script should give a clear failure.
        print(f"train_qlora.py failed: {exc}", file=sys.stderr)
        raise
