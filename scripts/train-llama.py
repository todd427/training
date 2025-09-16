#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QLoRA SFT for Llama 3.1 8B Instruct with chat templating + sequence packing,
driven by a fixed step budget and early stopping.

Install:
  pip install -U "transformers>=4.45" "datasets>=2.20" peft bitsandbytes accelerate evaluate

Data (JSONL per line):
  - {"messages":[{"role":"system|user|assistant","content":"..."}, ...]}
    (Your dataset already matches this.)
  - Or {"prompt":"...", "response":"..."} (auto-converted to messages).
"""

import os
import math
import argparse
from typing import Dict, List, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# --------- Utilities ---------
def normalize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure each example has a messages[] list in chat format."""
    if "messages" in rec and isinstance(rec["messages"], list):
        return {"messages": rec["messages"]}
    if "prompt" in rec and "response" in rec:
        return {
            "messages": [
                {"role": "user", "content": str(rec["prompt"])},
                {"role": "assistant", "content": str(rec["response"])},
            ]
        }
    raise ValueError("Record needs `messages` or (`prompt` and `response`).")


def apply_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def group_texts(examples: Dict[str, List[List[int]]], block_size: int) -> Dict[str, List[List[int]]]:
    """Concatenate token ids and split into fixed-size blocks (no padding)."""
    concat_ids = []
    for seq in examples["input_ids"]:
        concat_ids.extend(seq)
    total_len = (len(concat_ids) // block_size) * block_size
    concat_ids = concat_ids[:total_len]

    chunks = [concat_ids[i:i + block_size] for i in range(0, total_len, block_size)]
    masks = [[1] * block_size for _ in range(len(chunks))]
    return {"input_ids": chunks, "attention_mask": masks}


# --------- CLI ---------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct", type=str)
    ap.add_argument("--train_file", required=True, type=str)
    ap.add_argument("--val_file", required=True, type=str)
    ap.add_argument("--output_dir", default="ckpts/toddric-llama-8B-lora", type=str)

    # Training policy: fixed steps + early stopping
    ap.add_argument("--max_steps", default=1000, type=int)         # ~6 epochs on ~1.5k packed chunks
    ap.add_argument("--eval_steps", default=100, type=int)
    ap.add_argument("--save_steps", default=100, type=int)
    ap.add_argument("--early_stopping_patience", default=3, type=int)

    # Batch & sequence
    ap.add_argument("--batch_size", default=1, type=int)
    ap.add_argument("--grad_accum", default=24, type=int)
    ap.add_argument("--max_length", default=2048, type=int)        # 4096 if VRAM allows
    ap.add_argument("--no_packing", action="store_true")

    # Optim
    ap.add_argument("--lr", default=2e-4, type=float)
    ap.add_argument("--warmup_ratio", default=0.05, type=float)
    ap.add_argument("--scheduler", default="cosine", type=str, choices=["cosine", "linear"])

    # LoRA
    ap.add_argument("--lora_r", default=32, type=int)
    ap.add_argument("--lora_alpha", default=32, type=int)
    ap.add_argument("--lora_dropout", default=0.05, type=float)
    ap.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", type=str)

    # Precision
    ap.add_argument("--bf16", action="store_true")  # recommended on Ada/Hopper
    ap.add_argument("--seed", default=42, type=int)
    return ap.parse_args()


# --------- Main ---------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"[train_llama] Base model : {args.base_model}")
    print(f"[train_llama] Train file: {args.train_file}")
    print(f"[train_llama] Val file  : {args.val_file}")
    print(f"[train_llama] Out dir   : {args.output_dir}")
    print("[train_llama] QLoRA 4-bit: ON")

    # Quantization
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.chat_template is not None, "Tokenizer must include a chat_template (Llama 3.1)."

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
    )
    model.config.use_cache = False  # needed with gradient checkpointing

    # Prepare for k-bit + LoRA
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # Datasets
    raw_train = load_dataset("json", data_files=args.train_file, split="train")
    raw_val = load_dataset("json", data_files=args.val_file, split="train")

    def to_text(e):
        e = normalize_record(e)
        return {"text": apply_template(tokenizer, e["messages"])}

    train_text = raw_train.map(to_text, remove_columns=raw_train.column_names)
    val_text = raw_val.map(to_text, remove_columns=raw_val.column_names)

    # Tokenize (no extra specials; EOS boundaries already in template)
    def tok(batch):
        return tokenizer(batch["text"], add_special_tokens=False)
    train_tok = train_text.map(tok, batched=True, remove_columns=["text"])
    val_tok = val_text.map(tok, batched=True, remove_columns=["text"])

    # Packing
    if not args.no_packing:
        train_tok = train_tok.map(lambda x: group_texts(x, args.max_length), batched=True, batch_size=1000)
        val_tok = val_tok.map(lambda x: group_texts(x, args.max_length), batched=True, batch_size=1000)
    else:
        def pad_trunc(ex):
            out = tokenizer.pad(
                {"input_ids": ex["input_ids"]},
                padding="max_length", max_length=args.max_length, return_attention_mask=True
            )
            return out
        train_tok = train_tok.map(pad_trunc, batched=True)
        val_tok = val_tok.map(pad_trunc, batched=True)

    # Collator (labels=input_ids)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Precision flags
    use_bf16 = bool(args.bf16 and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup_ratio,
        logging_steps=max(10, args.eval_steps // 10),
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
    )
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    print(f"[info] packed train chunks={len(train_tok):,} | packed val chunks={len(val_tok):,}")
    est_steps_per_epoch = math.ceil(len(train_tok) / (args.batch_size * args.grad_accum))
    print(f"[info] rough steps/epoch≈{est_steps_per_epoch}")

    # Train
    trainer.train()
    trainer.save_state()
    trainer.save_model()  # saves the PEFT adapter

    # Eval + PPL
    metrics = trainer.evaluate()
    if "eval_loss" in metrics:
        try:
            metrics["perplexity"] = math.exp(metrics["eval_loss"])
        except OverflowError:
            metrics["perplexity"] = float("inf")
        print(f"[eval] loss={metrics['eval_loss']:.4f} | ppl={metrics['perplexity']:.2f}")
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    print("[done] Training complete.")


if __name__ == "__main__":
    main()

