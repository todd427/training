# LoRA + QLoRA Cheat-Sheet

## Core Ideas
- **LoRA**: Freeze base model, inject trainable low-rank matrices (rank = *r*) into linear layers. Trains adapters only.
- **QLoRA**: Load base weights in 4-bit (NF4 + double quant), keep compute in bf16/fp16, train LoRA on top.

---

## Data Prep
- Format: `{ "messages": [{"role": "system"|"user"|"assistant", "content": "..."}, ...] }`
- Render with `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)`
- Set: `tokenizer.pad_token = tokenizer.eos_token`

---

## Packing
- Concatenate tokenized samples, split into fixed blocks (`max_length=2048` or `4096`).
- Eliminates padding waste, improves throughput.

---

## QLoRA Config
```python
from transformers import BitsAndBytesConfig
bnb = BitsAndBytesConfig(
  load_in_4bit=True,
  bnb_4bit_quant_type="nf4",
  bnb_4bit_use_double_quant=True,
  bnb_4bit_compute_dtype=torch.bfloat16,
)
```

---

## LoRA Config
**Target modules (Llama 3.x):**
- `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`

**Capacity knob (r):**
- `r=32` → larger adapter (~84M trainable on 8B), expressive
- `r=16` → lighter (~42M trainable), often enough for style

```python
from peft import LoraConfig
lora = LoraConfig(
  r=32,
  lora_alpha=32,
  lora_dropout=0.05,
  bias="none",
  target_modules=[...],
  task_type="CAUSAL_LM",
)
```

---

## Training Arguments (sane defaults)
```python
TrainingArguments(
  output_dir="ckpts/toddric-llama-8B-lora",
  per_device_train_batch_size=1,
  gradient_accumulation_steps=24,
  max_steps=1000,                 # steps > epochs for small data
  learning_rate=2e-4,
  lr_scheduler_type="cosine",
  warmup_ratio=0.05,
  evaluation_strategy="steps",
  eval_steps=100,
  save_strategy="steps",
  save_steps=100,
  save_total_limit=2,
  load_best_model_at_end=True,
  metric_for_best_model="eval_loss",
  bf16=True,
  gradient_checkpointing=True,
  report_to=[],
)
```
Optional: `weight_decay=0.05`, `max_grad_norm=1.0`

---

## Signs of Progress
- **Loss drop**: expect eval loss ~1.2–1.5 (ppl ≈ 3.3–4.5) → useful adapter.
- **Eval plateau**: stop when eval loss flatlines/rises for 2–3 evals.
- **Outputs**: tuned model shows stylistic shift (tone, formatting, compliance).

---

## Pitfalls
- Forgetting chat template → nonsense training.
- Not setting pad_token → padding issues.
- No packing → wasted compute.
- Forgetting `use_cache=False` → checkpointing errors.
- Saving/merging wrong adapter → corrupted checkpoints.

---

## Deployment
- **Adapter inference**: efficient, small file.
- **Merged model**: full weights with adapter baked in (larger, but portable).

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16, device_map="auto")
peft_wrapped = PeftModel.from_pretrained(base, "ckpts/toddric-llama-8B-lora")
merged = peft_wrapped.merge_and_unload()
merged.save_pretrained("ckpts/toddric-llama-8B-lora/merged")
```

---

## Eval Workflow (picking the best checkpoint)
1. **Prepare prompts**: 5–10 short, representative “Todd tasks” (email rewrite, critique, summary).
2. **Deterministic pass**: run greedy decoding once to compare structure and instruction-following without sampling noise.

```bash
python scripts/eval_sxs_llama.py \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --ft_adapter ckpts/toddric-llama-8B-lora \
  --bf16 --greedy --max_new 220
```

3. **Stylistic pass**: enable sampling to judge tone and voice.

```bash
python scripts/eval_sxs_llama.py \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --ft_adapter ckpts/toddric-llama-8B-lora \
  --bf16 --max_new 220
```

4. **Quantitative checks**: track `eval_loss`/perplexity from training logs; lower is better until it plateaus.
5. **Leak check**: scan outputs for long verbatim spans from your training data. If present, you’re overfitting—prefer an earlier checkpoint.
6. **Choose checkpoint**: pick the one with the **best eval loss** that also reads most “Toddric” in side-by-side outputs.
7. **Archive metadata**: record commit hash, training args, step number, and a few example prompts/outputs alongside the chosen checkpoint.

---

## Quick Checklist (one-glance)
- [ ] Data → `messages` rendered via `apply_chat_template`
- [ ] `tokenizer.pad_token = tokenizer.eos_token`
- [ ] Packing enabled (2048/4096)
- [ ] QLoRA: NF4 + double quant; compute bf16/fp16
- [ ] LoRA targets: q/k/v/o + gate/up/down; `r=32` (try `r=16` later)
- [ ] Steps + early stopping; eval/save every 50–100 steps
- [ ] Side-by-side eval shows stronger Toddric tone/structure
- [ ] Archive best checkpoint + metadata (args, commit, step, samples)

