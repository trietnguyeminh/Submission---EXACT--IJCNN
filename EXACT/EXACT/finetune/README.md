# EXACT Physics Corrector Fine-tuning

This folder contains the fine-tuning workflow for the Type 2 physics pipeline.
The adapter is trained as a corrector/fallback: it decides whether to accept the
deterministic solver output or emit a corrected final answer.

## Build Dataset

```bash
python finetune/prepare_sft_dataset.py \
  --gold data/Physics_Problems.csv \
  --pred outputs/test10.jsonl \
  --mismatches outputs/type2_eval_mismatches.csv \
  --kb-root knowledge_base/physics \
  --out-dir finetune/data
```

## Train on Linux GPU

```bash
conda create -n exact-ft python=3.10 -y
conda activate exact-ft
pip install -U pip
pip install -r finetune/requirements-ft.txt

python finetune/train_qlora.py \
  --config finetune/configs/qwen3_8b_qlora_24gb.yaml
```

## Inference with Adapter

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

## Evaluate

```bash
python finetune/eval_corrector.py \
  --pred outputs/test_ft_corrector.jsonl \
  --gold data/Physics_Problems.csv \
  --baseline-mismatches outputs/type2_eval_mismatches.csv \
  --mismatches-out outputs/type2_eval_mismatches_ft_corrector.csv
```
