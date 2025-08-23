#!/usr/bin/env python3
"""
LoRA SFT for Toddric on JSONL (system/messages/tags).
Version-compatible with older/newer transformers + trl.
"""

import os, json, math, argparse, random
from typing import List
from datasets import Dataset, concatenate_datasets
from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
                          DataCollatorForLanguageModeling)
from trl import SFTTrainer
from peft import LoraConfig

# ---------------- Args ----------------
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
    ap.add_argument("--sentinel", default=None)
    ap.add_argument("--target_modules", nargs="*", default=None)
    ap.add_argument("--save_merged", action="store_true")
    return ap.parse_args()

# ---------------- Data ----------------
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
    s = sum(weights); weights = [w/s for w in weights]
    sizes = [len(d) for d in dsets]
    total = sum(sizes)
    target = [max(0,int(total*w)) for w in weights]
    rng = random.Random(1234)
    parts=[]
    for d,want in zip(dsets,target):
        if want<=0 or len(d)==0: continue
        idx=[rng.randrange(0,len(d)) for _ in range(want)]
        parts.append(d.select(idx))
    return concatenate_datasets(parts) if parts else concatenate_datasets(dsets)

# ---------------- Train ----------------
def train():
    args = parse_args()
    random.seed(args.seed)

    sentinel = ""
    if args.sentinel:
        with open(args.sentinel,"r",encoding="utf-8") as f:
            sentinel=f.read().strip()+"\n\n"

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token

    # safer model load (no device_map)
    model_kwargs=dict(torch_dtype="auto", low_cpu_mem_usage=False)
    if args.use_flash_attn:
        try: model_kwargs["attn_implementation"]="flash_attention_2"
        except: pass
    try:
        model=AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    except TypeError:
        model_kwargs.pop("attn_implementation",None)
        model=AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    dsets=[load_jsonl_as_dataset(p) for p in args.train_jsonl]
    train_ds=weighted_mix(dsets,args.train_weights) if (args.train_weights and len(args.train_weights)==len(dsets)) else concatenate_datasets(dsets)
    train_ds=train_ds.shuffle(seed=args.seed)
    n=len(train_ds)
    eval_n=max(32,int(n*args.eval_holdout))
    eval_ds=train_ds.select(range(eval_n))
    train_ds=train_ds.select(range(eval_n,n))

    def format_example(ex):
        system=ex.get("system","")
        if sentinel: system=sentinel+system
        messages=ex["messages"]
        if messages[-1]["role"]!="assistant":
            messages.append({"role":"assistant","content":""})
        msgs=[]
        if system: msgs.append({"role":"system","content":system})
        msgs.extend(messages)
        if hasattr(tok,"apply_chat_template"):
            text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=False)
        else:
            text="\n".join([f"{m['role']}: {m['content']}" for m in msgs])
        return {"text":text}

    train_fmt=train_ds.map(format_example,remove_columns=train_ds.column_names,desc="format train")
    eval_fmt=eval_ds.map(format_example,remove_columns=eval_ds.column_names,desc="format eval")

    lora=LoraConfig(r=args.lora_r,lora_alpha=args.lora_alpha,lora_dropout=args.lora_dropout,
                    bias="none",task_type="CAUSAL_LM",target_modules=args.target_modules or None)

    # TrainingArguments shim
    from inspect import signature
    steps_per_epoch=max(1,math.ceil(len(train_fmt)/max(1,args.batch_size)/max(1,args.grad_accum)))
    tot_steps=int(args.epochs*steps_per_epoch) if args.max_steps<0 else args.max_steps
    ta_kwargs=dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1,args.batch_size//2),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=math.ceil(args.epochs) if args.max_steps<0 else 1,
        max_steps=tot_steps if args.max_steps>0 else -1,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=args.bf16, fp16=args.fp16, tf32=args.tf32,
        gradient_checkpointing=args.gradient_checkpointing,
        lr_scheduler_type="cosine", optim="adamw_torch",
    )
    sig=signature(TrainingArguments.__init__)
    if "evaluation_strategy" in sig.parameters:
        ta_kwargs.update({"evaluation_strategy":"steps","eval_steps":args.eval_steps,
                          "report_to":[args.report_to] if args.report_to!="none" else []})
    elif "evaluate_during_training" in sig.parameters:
        ta_kwargs.update({"evaluate_during_training":True,"eval_steps":args.eval_steps})
    else:
        ta_kwargs.update({"do_eval":True})
    targs=TrainingArguments(**ta_kwargs)

    # SFTTrainer shim
    from inspect import signature as _sig
    def _formatting_func(batch): return batch["text"]
    sft_params=_sig(SFTTrainer.__init__).parameters
    sft_kwargs=dict(model=model,train_dataset=train_fmt,eval_dataset=eval_fmt,
                    peft_config=lora,data_collator=DataCollatorForLanguageModeling(tok,mlm=False),args=targs)
    if "tokenizer" in sft_params: sft_kwargs["tokenizer"]=tok
    if "dataset_text_field" in sft_params: sft_kwargs["dataset_text_field"]="text"
    elif "formatting_func" in sft_params: sft_kwargs["formatting_func"]=_formatting_func
    if "max_seq_length" in sft_params: sft_kwargs["max_seq_length"]=args.max_seq_len
    if "packing" in sft_params: sft_kwargs["packing"]=args.packing

    trainer=SFTTrainer(**sft_kwargs)
    trainer.train()

    trainer.model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)

    if args.save_merged:
        try:
            from peft import PeftModel
            base=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype="auto")
            merged=PeftModel.from_pretrained(base,args.output_dir)
            merged=merged.merge_and_unload()
            merged.save_pretrained(os.path.join(args.output_dir,"merged"))
        except Exception as e:
            print(f"[warn] merge failed: {e}")

if __name__=="__main__":
    train()
