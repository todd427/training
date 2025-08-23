#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA SFT for Toddric — with reporting & ETA.

Adds:
- --report_json <path> (default: <output_dir>/run_report.json)
- Pre-train dataset sanity report (counts, tags, empty-assistant %, length stats, est. truncation)
- ETA printouts during training

CLI example (same as you ran):
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

import os, json, math, argparse, random, time, statistics
from typing import List, Dict, Any, Tuple
from datasets import Dataset, concatenate_datasets
from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainingArguments, TrainerCallback)
from transformers import DataCollatorForLanguageModeling
from trl import SFTTrainer
from peft import LoraConfig

# -------------------- args --------------------
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
    ap.add_argument("--report_json", default=None, help="Where to write run_report.json (default inside output_dir)")
    return ap.parse_args()

# -------------------- IO --------------------
def load_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def load_jsonl_as_dataset(path: str) -> Tuple[Dataset, Dict[str,int], int]:
    """Return HF Dataset + tag_counts + empty_assistant_count (from raw)."""
    rows = []
    tag_counts = {}
    empty_assistant = 0
    for rec in load_lines(path):
        tags = rec.get("tags") or []
        for t in tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1
        msgs = rec.get("messages") or []
        if not msgs or msgs[0].get("role") != "user":
            continue
        # count genuinely empty assistants in raw (if present)
        if len(msgs) >= 2 and msgs[-1].get("role") == "assistant":
            if not (msgs[-1].get("content") or "").strip():
                empty_assistant += 1
        rows.append({
            "system": rec.get("system",""),
            "messages": msgs,
            "tags": tags
        })
    return Dataset.from_list(rows), tag_counts, empty_assistant

def weighted_mix(dsets: List[Dataset], weights: List[float]) -> Dataset:
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

# -------------------- ETA callback --------------------
class SimpleETACallback(TrainerCallback):
    def __init__(self, total_steps: int, log_every: int):
        self.total = max(1, total_steps)
        self.log_every = max(1, log_every)
        self.start = time.time()
        self.last_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step == 0 or (step % self.log_every) != 0:
            return
        now = time.time()
        elapsed = now - self.start
        rate = step / elapsed if elapsed > 0 else 0.0
        remaining = (self.total - step) / rate if rate > 0 else float("inf")
        def fmts(t):
            if not (t < 9e6): return "--:--:--"
            m, s = divmod(int(t), 60)
            h, m = divmod(m, 60)
            return f"{h:02d}:{m:02d}:{s:02d}"
        print(f"⏱  step {step}/{self.total} | {rate:.2f} it/s | ETA {fmts(remaining)}")

# -------------------- main --------------------
def main():
    args = parse_args()
    random.seed(args.seed)

    # Sentinel (guardrails)
    sentinel_text = None
    if args.sentinel:
        with open(args.sentinel, "r", encoding="utf-8") as f:
            sentinel_text = f.read().strip()

    # Tokenizer / model
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
        model.config.use_cache = False  # avoids warning

    # Load datasets + collect raw stats
    raw_tag_totals = {}
    raw_empty_assistants = 0
    dsets, paths = [], []
    for p in args.train_jsonl:
        ds, tag_counts, empty_cnt = load_jsonl_as_dataset(p)
        dsets.append(ds); paths.append(p)
        for k,v in tag_counts.items():
            raw_tag_totals[k] = raw_tag_totals.get(k, 0) + v
        raw_empty_assistants += empty_cnt

    if args.train_weights and len(args.train_weights) == len(dsets):
        train_ds = weighted_mix(dsets, args.train_weights)
        weight_info = {os.path.basename(p): w for p,w in zip(paths, args.train_weights)}
    else:
        train_ds = concatenate_datasets(dsets)
        weight_info = {os.path.basename(p): 1.0 for p in paths}

    # Shuffle + eval split
    train_ds = train_ds.shuffle(seed=args.seed)
    n_total = len(train_ds)
    eval_n = max(32, int(n_total * args.eval_holdout))
    eval_ds = train_ds.select(range(eval_n))
    train_ds = train_ds.select(range(eval_n, n_total))

    # Format via chat template (+ sentinel)
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

    # --------- Pre-train sanity report ---------
    # Sample a subset for length stats
    sample_n = min(1000, len(train_fmt))
    sample = train_fmt.select(range(sample_n))
    lens = [len(tok(x["text"]).input_ids) for x in sample]
    def pct(p): 
        k = max(0, min(len(lens)-1, int(round(p*(len(lens)-1)))))
        return sorted(lens)[k]
    mean_len = round(statistics.mean(lens), 1) if lens else 0.0
    p50 = pct(0.50) if lens else 0
    p95 = pct(0.95) if lens else 0
    p99 = pct(0.99) if lens else 0
    trunc_est = round(100.0 * sum(1 for L in lens if L > args.max_seq_len) / len(lens), 1) if lens else 0.0

    report = {
        "model": args.model,
        "output_dir": args.output_dir,
        "train_files": [os.path.abspath(p) for p in paths],
        "weights": weight_info,
        "counts": {
            "total_samples": n_total,
            "train_samples": len(train_fmt),
            "eval_samples": len(eval_fmt),
        },
        "raw_tag_totals": raw_tag_totals,
        "raw_empty_assistant_pairs": int(raw_empty_assistants),
        "seq_len_stats_on_sample": {
            "sample_n": sample_n,
            "mean": mean_len,
            "p50": p50, "p95": p95, "p99": p99,
            "max_seq_len": args.max_seq_len,
            "est_pct_truncated": trunc_est
        },
        "flags": {
            "bf16": bool(args.bf16),
            "fp16": bool(args.fp16),
            "tf32": bool(args.tf32),
            "packing": bool(args.packing),
            "grad_checkpointing": bool(args.gradient_checkpointing),
            "flash_attn": bool(args.use_flash_attn),
        }
    }

    os.makedirs(args.output_dir, exist_ok=True)
    report_path = args.report_json or os.path.join(args.output_dir, "run_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    # Pretty console dump
    print("=== Data sanity ===")
    print(json.dumps(report, indent=2))

    # LoRA
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules or None
    )

    steps_per_epoch = max(1, math.ceil(len(train_fmt) / (args.batch_size * args.grad_accum)))
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
        callbacks=[SimpleETACallback(total_steps=tot_steps if tot_steps>0 else steps_per_epoch, log_every=args.logging_steps)],
    )

    print(f"🔧 Samples: train={len(train_fmt)} eval={len(eval_fmt)} | steps/epoch={steps_per_epoch} | max_steps={tot_steps} | seq_len={args.max_seq_len}")
    t0 = time.time()
    trainer.train()
    wall = round(time.time() - t0, 1)

    # Save adapters + tokenizer
    trainer.model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)

    # Append final summary to report
    try:
        metrics = trainer.state.log_history[-1] if trainer.state.log_history else {}
    except Exception:
        metrics = {}
    summary = {
        "finished": True,
        "wall_seconds": wall,
        "final_global_step": int(getattr(trainer.state, "global_step", 0)),
        "final_train_loss": metrics.get("loss", None),
        "final_eval_loss": metrics.get("eval_loss", None),
        "output_dir": args.output_dir
    }
    with open(report_path, "r+", encoding="utf-8") as f:
        data = json.load(f); data["final"] = summary; f.seek(0); json.dump(data, f, ensure_ascii=False, indent=2); f.truncate()
    print(f"✅ Done. Report → {report_path}")

    # Optional: save merged (big)
    if args.save_merged:
        try:
            from peft import PeftModel
            base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
            merged = PeftModel.from_pretrained(base, args.output_dir).merge_and_unload()
            merged.save_pretrained(os.path.join(args.output_dir, "merged"))
        except Exception as e:
            print(f"[warn] merge failed: {e}")

if __name__ == "__main__":
    main()
