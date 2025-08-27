#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA/QLoRA training for Qwen2.5-3B-Instruct on SFT JSONL (messages[] schema).

Key features:
- Optional QLoRA (4-bit) via BitsAndBytes (use --no-4bit to disable)
- Left padding, eager attention, sliding-window disabled
- Assistant-only loss via offset_mapping (fast tokenizer required)
- Robust to HF version differences (eval_strategy vs evaluation_strategy)
- Accelerate unwrap_model() shim for older accelerate
- remove_unused_columns=False so raw rows reach our collator
- Num workers configurable (default 0 for clarity)
"""

import os, re, json, math, inspect, argparse
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    Trainer, TrainingArguments
)

# ---- Optional 4-bit quantization ----
try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    BitsAndBytesConfig = None
    _HAS_BNB = False

# ---- PEFT / LoRA ----
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---- Accelerate compatibility shim (unwrap_model kwarg) ----
try:
    from accelerate import Accelerator
    if "keep_torch_compile" not in inspect.signature(Accelerator.unwrap_model).parameters:
        _orig_unwrap = Accelerator.unwrap_model
        def _unwrap_shim(self, model, **kwargs):
            # drop unknown kwargs for older accelerate versions
            return _orig_unwrap(self, model)
        Accelerator.unwrap_model = _unwrap_shim
except Exception:
    pass

# ---------------- utils

def pick_dtype():
    """bfloat16 on Ampere+; else float16; CPU fallback float32."""
    if torch.cuda.is_available():
        major = torch.cuda.get_device_capability(0)[0]
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32

def canon(s: str) -> str:
    s = (s or "").replace("\r\n","\n").replace("\r","\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

# ---------------- data

class JsonlDataset(Dataset):
    def __init__(self, path):
        self.rows=[]
        path = os.path.expanduser(path)
        with open(path, encoding="utf-8") as f:
            for ln in f:
                try:
                    o=json.loads(ln)
                except Exception:
                    continue
                if isinstance(o, dict) and isinstance(o.get("messages"), list) and len(o["messages"])>=2:
                    self.rows.append(o)
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

def build_labels_for_assistant(tok, text, messages, enc):
    """
    Construct labels that train only on assistant spans.
    Uses offset_mapping to map assistant character spans → token indices.
    """
    spans=[]; search_from=0
    for m in messages:
        if m.get("role")!="assistant": continue
        content = canon(m.get("content",""))
        if not content: continue
        idx = text.find(content, search_from)
        if idx==-1:
            # fuzzy whitespace collapse match
            T = re.sub(r"\s+", " ", text[search_from:])
            C = re.sub(r"\s+", " ", content)
            j = T.find(C)
            if j!=-1: idx = search_from + j
            else:     continue
        s, e = idx, idx+len(content)
        spans.append((s,e))
        search_from = e

    labels = torch.full_like(enc["input_ids"], -100)
    offs = enc["offset_mapping"][0].tolist()  # List[(start,end)]
    for i,(a,b) in enumerate(offs):
        if a==b==0:  # many fast tokenizers use (0,0) for special/pad
            continue
        for (s,e) in spans:
            if not (b<=s or a>=e):   # overlap
                labels[0, i] = enc["input_ids"][0, i]
                break
    return labels

class ChatCollator:
    """Offsets-only collator (reliable across HF versions)."""
    def __init__(self, tok, max_len):
        self.tok = tok
        self.max_len = max_len
        if not getattr(tok, "is_fast", False):
            raise RuntimeError("This collator requires a *fast* tokenizer (offset_mapping). Re-load with use_fast=True.")

    def __call__(self, batch):
        input_ids=[]; attn=[]; labels=[]
        good=0; bad=0

        for ex in batch:
            ex_id = ex.get("id") if isinstance(ex, dict) else None
            if not isinstance(ex, dict) or "messages" not in ex or not isinstance(ex["messages"], list) or not ex["messages"]:
                print(f"[collate:skip] bad sample. id={ex_id}", flush=True); bad+=1; continue

            msgs = ex["messages"]
            # ensure system first
            if msgs[0].get("role")!="system":
                msgs = [{"role":"system","content":"You are a helpful assistant."}] + msgs
            # canonicalize
            msgs = [{"role":m.get("role"), "content":canon(m.get("content",""))} for m in msgs]

            # 1) Build full chat text (templated string)
            text = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )

            # 2) Tokenize with offsets
            enc = self.tok(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_len,
                return_offsets_mapping=True,
            )
            if "offset_mapping" not in enc:
                print(f"[collate:skip] no offset_mapping. id={ex_id}", flush=True); bad+=1; continue

            # 3) Assistant-only labels by span overlap
            lab = build_labels_for_assistant(self.tok, text, msgs, enc)

            input_ids.append(enc["input_ids"][0])
            attn.append(enc["attention_mask"][0])
            labels.append(lab[0])
            good += 1

        if good == 0:
            # Last-resort: no-op batch to avoid crash; prints once and continues
            from torch import tensor
            print(f"[collate:warn] All {len(batch)} samples invalid (bad={bad}). Inserting no-op batch.", flush=True)
            dummy = self.tok("", return_tensors="pt")
            batch_enc = {"input_ids": dummy["input_ids"],
                         "attention_mask": dummy["attention_mask"],
                         "labels": tensor([[-100]], dtype=torch.long)}
            return batch_enc

        # 4) Left-pad & align labels
        batch_enc = self.tok.pad({"input_ids": input_ids, "attention_mask": attn},
                                 padding=True, return_tensors="pt")
        maxlen = batch_enc["input_ids"].size(1)

        # enforce integer dtypes expected by embedding / CE loss
        batch_enc["input_ids"] = batch_enc["input_ids"].to(dtype=torch.long)

        padded=[]
        for lab in labels:
            pad = maxlen - lab.size(0)
            if pad>0:
                lab = torch.nn.functional.pad(lab, (pad,0), value=-100)
            padded.append(lab)
        batch_enc["labels"] = torch.stack(padded, dim=0).to(dtype=torch.long)
        return batch_enc

# ---------------- model

def load_base(model_id, use_4bit=True):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    cfg = AutoConfig.from_pretrained(model_id)
    if getattr(cfg, "sliding_window", None):
        cfg.sliding_window = None

    quant_cfg = None
    if use_4bit and _HAS_BNB:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=pick_dtype(),
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=pick_dtype(),
        device_map="auto",
        attn_implementation="eager",
        quantization_config=quant_cfg,
    )
    if getattr(model.generation_config, "sliding_window", None):
        model.generation_config.sliding_window = None
    model.config.use_cache = False
    if quant_cfg:
        model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    return tok, model

def discover_target_modules(model):
    names=set()
    for n,_ in model.named_modules():
        if any(k in n for k in ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","W_pack"]):
            names.add(n.split(".")[-1])
    return sorted(names) if names else ["W_pack","o_proj","gate_proj","up_proj","down_proj"]

def build_peft(model, r=32, alpha=64, dropout=0.05, target_modules=None):
    if target_modules is None:
        target_modules = discover_target_modules(model)
    lcfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    return get_peft_model(model, lcfg)

def make_training_args(eval_steps, **kw):
    """
    Transformers compatibility:
    - If TrainingArguments has 'eval_strategy', use that.
    - Else if it has 'evaluation_strategy', use that.
    - Else omit strategy (defaults to 'no').
    """
    params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    if "eval_strategy" in params:
        return TrainingArguments(eval_strategy="steps", eval_steps=eval_steps, **kw)
    elif "evaluation_strategy" in params:
        return TrainingArguments(evaluation_strategy="steps", eval_steps=eval_steps, **kw)
    else:
        return TrainingArguments(**kw)

# ---------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--train", default=os.path.expanduser("~/training/data/sft_ready/train.jsonl"))
    ap.add_argument("--val",   default=os.path.expanduser("~/training/data/sft_ready/val.jsonl"))
    ap.add_argument("--out",   default=os.path.expanduser("~/training/ckpts/toddric-3b-lora-v1"))
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--mbatch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--eval_steps", type=int, default=250)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no-4bit", action="store_true", help="Disable 4-bit QLoRA (run BF16/FP16 LoRA)")
    ap.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 recommended for debugging)")
    args = ap.parse_args()

    # perf toggles
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    use_4bit = (not args.no_4bit) and _HAS_BNB
    if not _HAS_BNB and not args.no_4bit:
        print("[warn] bitsandbytes not found; training without 4-bit quantization.")

    tok, base = load_base(args.base, use_4bit=use_4bit)
    model = build_peft(base, r=args.r, alpha=args.alpha, dropout=args.dropout)

    train_ds = JsonlDataset(args.train)
    val_ds   = JsonlDataset(args.val)
    collate  = ChatCollator(tok, args.seq)

    steps_per_epoch = math.ceil(len(train_ds) / (args.mbatch * args.accum))
    print(f"[info] train rows={len(train_ds)} | val rows={len(val_ds)} | steps/epoch≈{steps_per_epoch}")

    # TrainingArguments (remove_unused_columns=False so our raw dict reaches collator)
    common = dict(
        output_dir=args.out,
        per_device_train_batch_size=args.mbatch,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        weight_decay=args.wd,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_steps=50,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=(pick_dtype()==torch.bfloat16),
        fp16=(pick_dtype()==torch.float16),
        dataloader_pin_memory=True,
        dataloader_num_workers=args.num_workers,
        gradient_checkpointing=True,
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
    )
    targs = make_training_args(args.eval_steps, **common)

    # Trainer init: prefer processing_class if supported (tokenizer deprecation)
    trainer_kwargs = dict(model=model, args=targs, train_dataset=train_ds,
                          eval_dataset=val_ds, data_collator=collate)
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tok
    else:
        trainer_kwargs["tokenizer"] = tok

    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"[done] LoRA saved → {args.out}")

if __name__ == "__main__":
    main()
