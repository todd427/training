#!/usr/bin/env python3
"""
LoRA SFT for Toddric on JSONL (system/messages/tags).

Each JSONL line:
{
  "system": "...",
  "messages": [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}],
  "tags": ["epub","summary","v1"]
}

Example:
python sft_train.py \
  --model HuggingFaceH4/zephyr-7b-beta \
  --train_jsonl out/sent.sample1k.sft.jsonl out/memories.sample1k.sft.jsonl out/epub.sample1k.sft.jsonl out/chatgpt.sample1k.sft.jsonl \
  --train_weights 0.3 0.2 0.2 0.3 \
  --output_dir ./ckpts/toddric-smoke \
  --epochs 1 --lr 2e-5 --batch_size 2 --grad_accum 24 \
  --max_seq_len 2048 --bf16 --packing --gradient_checkpointing --use_flash_attn \
  --max_steps 600 --eval_steps 200 --save_steps 600 \
  --sentinel ./sentinel_guardrails.txt
"""

import os, json, math, argparse, random
from typing import List, Dict, Any
from datasets import Dataset, concatenate_datasets
from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainingArguments)
from trl import SFTTrainer
from peft import LoraConfig
from transformers import DataCollatorForLanguageModeling

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train_jsonl", nargs="+", required=True)
    ap.add_argument("--train_weights", nargs="+", type=float, default=None)
    ap.add_argument("--eval_holdout", type=float, default=0.02)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--packing", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--logging_steps", type=int, default=25)
    ap.add_argument("--eval_steps", type=int, default=1000)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--use_flash_attn", action="store_true")
    ap.add_argument("--report_to", default="none")
    ap.add_argument("--wandb_project", default="toddric")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", nargs="*", default=None)
    ap.add_argument("--save_merged", action="store_true")
    ap.add_argument("--sentinel", default=None, help="Text file with guardrails to prepend to system prompts")
    return ap.parse_args()

def load_jsonl_as_dataset(path: str) -> Dataset:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            msgs = rec.get("messages") or []
            if not msgs or msgs[0].get("role") != "user":
                continue
            rows.append({
                "system": rec.get("system",""),
                "messages": msgs,
                "tags": rec.get("tags",[])
            })
    return Dataset.from_list(rows)

def weighted_mix(dsets: List[Dataset], weights: List[float]) -> Dataset:
    # Normalize weights and sample-with-replacement to approximate ratios
    s = sum(weights); weights = [w/s for w in weights]
    sizes = [len(d) for d in dsets]
    total = sum(sizes)
    target = [max(1, int(total * w)) for w in weights]
    rng = random.Random(1234)
    parts = []
    for d, want in zip(dsets, target):
        if len(d) == 0: continue
        idx = [rng.randrange(0, len(d)) for _ in range(want)]
        parts.append(d.select(idx))
    return concatenate_datasets(parts) if parts else concatenate_datasets(dsets)

def main():
    args = parse_args()
    random.seed(args.seed)

    # Load sentinel (guardrails) if provided
    sentinel_text = None
    if args.sentinel:
        with open(args.sentinel, "r", encoding="utf-8") as f:
            sentinel_text = f.read().strip()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    attn_impl = "flash_attention_2" if args.use_flash_attn else "eager"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation=attn_impl if args.use_flash_attn else None
    )
    if args.gradient_checkpointing:
        model.config.use_cache = False  # avoid warning

    # Load datasets
    dsets = [load_jsonl_as_dataset(p) for p in args.train_jsonl]
    if args.train_weights and len(args.train_weights) == len(dsets):
        train_ds = weighted_mix(dsets, args.train_weights)
    else:
        train_ds = concatenate_datasets(dsets)

    # Shuffle + eval split
    train_ds = train_ds.shuffle(seed=args.seed)
    n = len(train_ds)
    eval_n = max(32, int(n * args.eval_holdout))
    eval_ds = train_ds.select(range(eval_n))
    train_ds = train_ds.select(range(eval_n, n))

    # Format through chat template, injecting sentinel
    def format_example(ex):
        system = ex.get("system","")
        if sentinel_text:
            system = (sentinel_text + ("\n\n" + system if system else ""))
        messages = ex["messages"]
        if messages[-1]["role"] != "assistant":
            messages.append({"role":"assistant","content":""})
        msgs = []
        if system:
            msgs.append({"role":"system","content":system})
        msgs.extend(messages)
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    train_fmt = train_ds.map(format_example, remove_columns=train_ds.column_names, desc="format train")
    eval_fmt  = eval_ds.map(format_example,  remove_columns=eval_ds.column_names,  desc="format eval")

    # LoRA
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules or None
    )

    steps_per_epoch = math.ceil(len(train_fmt) / (args.batch_size * args.grad_accum)) or 1
    tot_steps = int(args.epochs * steps_per_epoch) if args.max_steps < 0 else args.max_steps

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size // 2),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=math.ceil(args.epochs) if args.max_steps < 0 else 1,
        max_steps=tot_steps if args.max_steps > 0 else -1,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=args.bf16,
        fp16=args.fp16,
        tf32=args.tf32,
        gradient_checkpointing=args.gradient_checkpointing,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        report_to=[args.report_to] if args.report_to != "none" else [],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=train_fmt,
        eval_dataset=eval_fmt,
        peft_config=lora,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=args.packing,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
        args=targs,
    )

    print(f"🔧 Samples: train={len(train_fmt)} eval={len(eval_fmt)} | max_steps={tot_steps} | seq_len={args.max_seq_len}")
    trainer.train()

    # Save adapters
    trainer.model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)

    if args.save_merged:
        try:
            from peft import PeftModel
            base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
            merged = PeftModel.from_pretrained(base, args.output_dir)
            merged = merged.merge_and_unload()
            merged.save_pretrained(os.path.join(args.output_dir, "merged"))
        except Exception as e:
            print(f"[warn] merge failed: {e}")

if __name__ == "__main__":
    main()
