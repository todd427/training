# Training: Toddric LoRA/QLoRA Fine-Tuning

This repository contains scripts and configs for fine-tuning **Llama-3.1-8B-Instruct** with LoRA adapters using QLoRA (4-bit quantization).

## Requirements

```bash
pip install -U transformers accelerate datasets peft bitsandbytes evaluate
# optional: flash-attn (GPU/driver dependent)
```

## Data Format

Each line of your JSONL should look like:

```json
{
  "messages": [
    {"role": "system", "content": "You are Toddric, Todd’s helpful assistant."},
    {"role": "user", "content": "Write an email in my voice."},
    {"role": "assistant", "content": "Here’s the rewritten email..."}
  ]
}
```

The training script applies Llama-3.1’s native chat template.

## Training

Main script: `scripts/train_llama.py`

Example run:

```bash
python scripts/train_llama.py \
  --train_file data/sft_ready/train.jsonl \
  --val_file   data/sft_ready/val.jsonl \
  --output_dir ckpts/toddric-llama-8B-lora \
  --bf16 --max_steps 1000 \
  --eval_steps 100 --save_steps 100 \
  --batch_size 1 --grad_accum 24 --max_length 2048
```

Key details:

* **QLoRA**: loads the base in 4-bit (NF4 + double quant), trains LoRA adapters in bf16/fp16.
* **Steps over epochs**: for small datasets (\~1.5k examples), `--max_steps` gives finer control.
* **Early stopping**: enabled, so training halts when eval loss stops improving.
* **Adapters only**: base weights are frozen; only LoRA inserts are trained.

## Evaluation

Compare base vs fine-tuned with side-by-side outputs:

```bash
python scripts/eval_sxs_llama.py \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --ft_adapter ckpts/toddric-llama-8B-lora \
  --bf16 --max_new 220
```

Outputs are saved as CSV/JSON in `eval_results_llama/`.

## Merge (optional)

You can merge the adapter into the base to get a single set of weights:

```bash
python scripts/train_llama.py \
  --train_file data/sft_ready/train.jsonl \
  --val_file data/sft_ready/val.jsonl \
  --output_dir ckpts/toddric-llama-8B-lora \
  --bf16 --max_steps 1000 --merge_and_save
```

Merged weights are saved under `ckpts/toddric-llama-8B-lora/merged`.

## Common Pitfalls

* **Chat template not applied** → model learns gibberish. Always use `apply_chat_template`.
* **PAD not set** → set `tokenizer.pad_token = tokenizer.eos_token`.
* **No packing** → wastes tokens; enable packing for 2048/4096 contexts.
* **Forget to disable use\_cache** → breaks gradient checkpointing.

## Next Steps

* **v1**: your first working adapter (r=32).
* **v2**: rerun with r=16 for lighter adapters or add more data.
* Keep README practical; see `CHEATSHEET.md` for detailed explanations and tuning advice.
# Training: Toddric LoRA/QLoRA Fine-Tuning

This repository contains scripts and configs for fine-tuning **Llama-3.1-8B-Instruct** with LoRA adapters using QLoRA (4-bit quantization).

## Requirements

```bash
pip install -U transformers accelerate datasets peft bitsandbytes evaluate
# optional: flash-attn (GPU/driver dependent)
```

## Data Format

Each line of your JSONL should look like:

```json
{
  "messages": [
    {"role": "system", "content": "You are Toddric, Todd’s helpful assistant."},
    {"role": "user", "content": "Write an email in my voice."},
    {"role": "assistant", "content": "Here’s the rewritten email..."}
  ]
}
```

The training script applies Llama-3.1’s native chat template.

## Training

Main script: `scripts/train_llama.py`

Example run:

```bash
python scripts/train_llama.py \
  --train_file data/sft_ready/train.jsonl \
  --val_file   data/sft_ready/val.jsonl \
  --output_dir ckpts/toddric-llama-8B-lora \
  --bf16 --max_steps 1000 \
  --eval_steps 100 --save_steps 100 \
  --batch_size 1 --grad_accum 24 --max_length 2048
```

Key details:

* **QLoRA**: loads the base in 4-bit (NF4 + double quant), trains LoRA adapters in bf16/fp16.
* **Steps over epochs**: for small datasets (\~1.5k examples), `--max_steps` gives finer control.
* **Early stopping**: enabled, so training halts when eval loss stops improving.
* **Adapters only**: base weights are frozen; only LoRA inserts are trained.

## Evaluation

Compare base vs fine-tuned with side-by-side outputs:

```bash
python scripts/eval_sxs_llama.py \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --ft_adapter ckpts/toddric-llama-8B-lora \
  --bf16 --max_new 220
```

Outputs are saved as CSV/JSON in `eval_results_llama/`.

## Merge (optional)

You can merge the adapter into the base to get a single set of weights:

```bash
python scripts/train_llama.py \
  --train_file data/sft_ready/train.jsonl \
  --val_file data/sft_ready/val.jsonl \
  --output_dir ckpts/toddric-llama-8B-lora \
  --bf16 --max_steps 1000 --merge_and_save
```

Merged weights are saved under `ckpts/toddric-llama-8B-lora/merged`.

## Common Pitfalls

* **Chat template not applied** → model learns gibberish. Always use `apply_chat_template`.
* **PAD not set** → set `tokenizer.pad_token = tokenizer.eos_token`.
* **No packing** → wastes tokens; enable packing for 2048/4096 contexts.
* **Forget to disable use\_cache** → breaks gradient checkpointing.

## Next Steps

* **v1**: your first working adapter (r=32).
* **v2**: rerun with r=16 for lighter adapters or add more data.
* Keep README practical; see `CHEATSHEET.md` for detailed explanations and tuning advice.

