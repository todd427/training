#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QLoRA SFT for Llama 3.1 8B Instruct with:
- native chat templating (tokenizer.apply_chat_template)
- sequence packing (concat + chunk to max_length)
- max_steps training + early stopping
- best-checkpoint saving and optional merge

Install (rec):
  pip install -U "transformers>=4.45" "datasets>=2.20" peft bitsandbytes accelerate evaluate

Optional speed-ups:
  - flash-attn (GPU/driver dependent)
"""

import os
import math
import argparse
from typing import Dict, List, Any

import torch
from datasets import load_dataset
from inspect import signature
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
)


# -------------------------
# Data helpers
# -------------------------
def normalize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure a 'messages' list exists. If only prompt/response, convert."""
    if "messages" in rec and isinstance(rec["messages"], list):
        return {"messages": rec["messages"]}
    if "prompt" in rec and "response" in rec:
        return {
            "messages": [
                {"role": "user", "content": str(rec["prompt"])},
                {"role": "assistant", "content": str(rec["response"])},
            ]
        }
    raise ValueError("Record missing 'messages' OR 'prompt'+'response' fields.")


def render_with_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    """Render a conversation into Llama 3.1's chat format."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,  # SFT: include targets
    )


def group_texts(examples: Dict[str, List[List[int]]], block_size: int) -> Dict[str, List[List[int]]]:
    """Pack: concatenate token lists and split into fixed blocks."""
    concatenated = []
    for ids in examples["input_ids"]:
        concatenated.extend(ids)

    total_length = (len(concatenated) // block_size) * block_size
    concatenated = concatenated[:total_length]

    result_ids = [concatenated[i: i + block_size] for i in range(0, total_length, block_size)]
    result_mask = [[1] * block_size for _ in range(len(result_ids))]
    return {"input_ids": result_ids, "attention_mask": result_mask}


# -------------------------
# Args
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser("train_llama.py")
    ap.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct", type=str)
    ap.add_argument("--train_file", required=True, type=str)
    ap.add_argument("--val_file", required=True, type=str)
    ap.add_argument("--output_dir", default="ckpts/toddric-llama-8B-lora", type=str)

    # Training schedule: prefer max_steps + early stopping
    ap.add_argument("--max_steps", default=1000, type=int, help="Total optimizer steps. If <=0, falls back to epochs.")
    ap.add_argument("--epochs", default=1, type=int, help="Used only if max_steps<=0.")
    ap.add_argument("--eval_steps", default=100, type=int)
    ap.add_argument("--save_steps", default=100, type=int)
    ap.add_argument("--logging_steps", default=10, type=int)
    ap.add_argument("--early_stopping_patience", default=3, type=int)

    # Model/optimizer knobs
    ap.add_argument("--bf16", action="store_true", help="Enable bfloat16 compute (recommended on Ada/Hopper).")
    ap.add_argument("--lr", default=2e-4, type=float)
    ap.add_argument("--warmup_ratio", default=0.05, type=float)
    ap.add_argument("--batch_size", default=1, type=int)
    ap.add_argument("--grad_accum", default=24, type=int)
    ap.add_argument("--max_length", default=2048, type=int, help="Packed sequence length.")
    ap.add_argument("--no_packing", action="store_true")

    # LoRA
    ap.add_argument("--lora_r", default=32, type=int)
    ap.add_argument("--lora_alpha", default=32, type=int)
    ap.add_argument("--lora_dropout", default=0.05, type=float)
    ap.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", type=str)

    # Misc
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--resume_from", default=None, type=str, help="Checkpoint dir to resume from.")
    ap.add_argument("--merge_and_save", action="store_true", help="Merge LoRA into base and save a full model.")
    return ap.parse_args()


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"[train_llama] Base model : {args.base_model}")
    print(f"[train_llama] Train file : {args.train_file}")
    print(f"[train_llama] Val file   : {args.val_file}")
    print(f"[train_llama] Output dir : {args.output_dir}")
    print(f"[train_llama] QLoRA 4-bit: ON")

    # Quantization (QLoRA)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.chat_template is not None, (
        "No chat template found. Ensure transformers is new enough and model id is correct."
    )

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        dtype=torch.bfloat16 if args.bf16 else torch.float16,  # use new kwarg to silence deprecation warning
        device_map="auto",
    )
    model.config.use_cache = False  # for gradient checkpointing
    model = prepare_model_for_kbit_training(model)

    # LoRA
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

    def to_text(ex):
        norm = normalize_record(ex)
        text = render_with_template(tokenizer, norm["messages"])
        return {"text": text}

    train_text = raw_train.map(to_text, remove_columns=raw_train.column_names)
    val_text = raw_val.map(to_text, remove_columns=raw_val.column_names)

    # Tokenize
    def tok(batch):
        # The chat template already includes BOS/EOS; don't add again.
        return tokenizer(batch["text"], add_special_tokens=False)

    train_tok = train_text.map(tok, batched=True, remove_columns=["text"])
    val_tok = val_text.map(tok, batched=True, remove_columns=["text"])

    # Packing
    if not args.no_packing:
        train_tok = train_tok.map(lambda x: group_texts(x, args.max_length), batched=True, batch_size=1000)
        val_tok = val_tok.map(lambda x: group_texts(x, args.max_length), batched=True, batch_size=1000)
    else:
        # Fallback: pad/trunc to fixed length (less efficient than packing)
        def pad_trunc(examples):
            outs = tokenizer.pad(
                {"input_ids": examples["input_ids"]},
                padding="max_length",
                max_length=args.max_length,
                return_attention_mask=True,
            )
            return outs

        train_tok = train_tok.map(pad_trunc, batched=True)
        val_tok = val_tok.map(pad_trunc, batched=True)

    # Collator (labels = input_ids for causal LM)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training args (version-agnostic eval[u]ation_strategy)
    use_bf16 = bool(args.bf16 and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8)

    ta_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        num_train_epochs=args.epochs if args.max_steps <= 0 else 1,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
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
    sig = signature(TrainingArguments.__init__)
    if "evaluation_strategy" in sig.parameters:
        ta_kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in sig.parameters:
        ta_kwargs["eval_strategy"] = "steps"
    # else: ultra-old fallback—Trainer will still eval because eval_steps>0

    training_args = TrainingArguments(**ta_kwargs)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
    )
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    # Info
    eff_train = len(train_tok)
    eff_val = len(val_tok)
    steps_per_epoch = math.ceil(eff_train / (args.batch_size * args.grad_accum)) if eff_train else 0
    print(f"[info] packed_train={eff_train:,} | packed_val={eff_val:,} | steps/epoch≈{steps_per_epoch}")

    # Train (resume handled here, not in TrainingArguments)
    train_result = trainer.train(resume_from_checkpoint=args.resume_from)
    trainer.save_state()
    trainer.save_model()  # saves the PEFT adapter

    # Eval (perplexity)
    metrics = trainer.evaluate()
    if "eval_loss" in metrics:
        try:
            metrics["perplexity"] = math.exp(metrics["eval_loss"])
        except OverflowError:
            metrics["perplexity"] = float("inf")
        print(f"[eval] loss={metrics['eval_loss']:.4f} | ppl={metrics['perplexity']:.2f}")
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # Optional merge
    if args.merge_and_save:
        print("[merge] Merging LoRA into base weights...")
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            dtype=torch.bfloat16 if args.bf16 else torch.float16,
            device_map="auto",
        )
        # Load trained adapter into base then merge
        peft_wrapped = PeftModel.from_pretrained(base, args.output_dir)
        merged = peft_wrapped.merge_and_unload()
        merged.save_pretrained(os.path.join(args.output_dir, "merged"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "merged"))
        print(f"[merge] Full model saved to {os.path.join(args.output_dir, 'merged')}")

    print("[done] Training complete.")


if __name__ == "__main__":
    main()

