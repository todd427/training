#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train LoRA/QLoRA on Qwen2 / Qwen2.5 with:
 - Robust dataset filtering (only rows with an assistant turn)
 - Batched fast-tokenizer path
 - Assistant-only labels parsed from Qwen chat tags
 - SafeTrainer that enforces integer IDs and computes CE loss if needed
 - BitsAndBytesConfig (no deprecated args), eval_strategy + processing_class

Run:
  python scripts/train_lora_qwen_faster.py \
    --base Qwen/Qwen2-1.5B-Instruct \
    --train ~/training/data/sft_ready/train.jsonl \
    --val   ~/training/data/sft_ready/val.jsonl \
    --out   ~/training/ckpts/toddric-1_5b-lora-v1 \
    --mbatch 2 --accum 8 --seq 2048 --epochs 2 \
    --debug_labels 50
"""

import os
import re
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    set_seed,
    BitsAndBytesConfig,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Optional 4-bit backend
try:
    import bitsandbytes as bnb  # noqa: F401
    HAS_BNB = True
except Exception:
    HAS_BNB = False


# ----------------- Helpers -----------------

def canon(x: Any) -> str:
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    return x.replace("\r\n", "\n").strip()


def to_messages(ex: Dict[str, Any]) -> List[Dict[str, str]]:
    """Normalize a row to messages[]. Return [] if we cannot."""
    if not isinstance(ex, dict):
        return []
    # Common schemas
    if isinstance(ex.get("messages"), list):
        msgs = ex["messages"]
    elif "prompt" in ex and "response" in ex:
        msgs = [
            {"role": "system", "content": "You are a helpful, concise assistant."},
            {"role": "user", "content": canon(ex["prompt"])},
            {"role": "assistant", "content": canon(ex["response"])},
        ]
    elif isinstance(ex.get("conversations"), list):  # ShareGPT-like
        conv = []
        for turn in ex["conversations"]:
            role = turn.get("from") or turn.get("role")
            if role == "human" or role == "user":
                role = "user"
            elif role == "gpt" or role == "assistant":
                role = "assistant"
            elif role == "system":
                role = "system"
            else:
                continue
            conv.append({"role": role, "content": canon(turn.get("value") or turn.get("content") or "")})
        msgs = conv
    elif "instruction" in ex and "output" in ex:  # Alpaca-ish
        inp = canon(ex.get("input", ""))
        prompt = canon(ex["instruction"]) + (("\n" + inp) if inp else "")
        msgs = [
            {"role": "system", "content": "You are a helpful, concise assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": canon(ex["output"])},
        ]
    else:
        msgs = []

    # Canonicalize + ensure roles
    msgs = [{"role": m.get("role"), "content": canon(m.get("content"))}
            for m in msgs if isinstance(m, dict) and m.get("role") and m.get("content") is not None]
    return msgs


def ensure_system_first(msgs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not msgs or msgs[0].get("role") != "system":
        msgs = [{"role": "system", "content": "You are a helpful, concise assistant."}] + msgs
    return msgs


# Qwen tag regexes (DOTALL to span lines). We capture only the message content in group(1).
_PATTERNS = [
    re.compile(r"<\|im_start\|>\s*assistant\s*\n(.*?)<\|im_end\|>", re.DOTALL),  # Qwen2/Qwen2.5
    re.compile(r"<\|assistant\|>(.*?)</\|assistant\|>", re.DOTALL),              # alternate
    re.compile(r"<\|assistant\|>\s*(.+)$", re.DOTALL),                           # single-start tag
]

def find_assistant_char_spans_from_tags(templated_text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for pat in _PATTERNS:
        for m in pat.finditer(templated_text):
            spans.append(m.span(1))
    if not spans:
        return []
    # merge overlaps
    spans.sort()
    merged = [spans[0]]
    for s, e in spans[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def build_labels_for_assistant(text: str, enc: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Label only tokens overlapping assistant spans; others = -100.
    """
    input_ids = enc["input_ids"][0]        # [T]
    offsets   = enc["offset_mapping"][0]   # [T, 2]
    T = input_ids.size(0)
    labels = torch.full((T,), fill_value=-100, dtype=torch.long)

    spans = find_assistant_char_spans_from_tags(text)
    if not spans:
        return labels.unsqueeze(0)

    for i in range(T):
        s, e = offsets[i].tolist()
        if s == e == 0:
            continue  # special/pad
        for as_s, as_e in spans:
            if not (e <= as_s or s >= as_e):  # overlap
                labels[i] = input_ids[i]
                break
    return labels.unsqueeze(0)  # [1, T]


# ----------------- Dataset -----------------

class JsonlChatDataset(Dataset):
    def __init__(self, path: str):
        self.rows = []
        path = os.path.expanduser(path)
        total = 0
        valid = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                total += 1
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                msgs = to_messages(ex)
                # keep only rows that have at least one assistant turn
                if msgs and any(m.get("role") == "assistant" and canon(m.get("content")) for m in msgs):
                    self.rows.append({"messages": msgs, "id": ex.get("id")})
                    valid += 1
        print(f"[dataset] {os.path.basename(path)}: kept {valid}/{total} rows with an assistant turn.")
        if valid == 0:
            raise RuntimeError(f"No valid rows with assistant content in {path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


# ----------------- Collator (batched fast path + diagnostics) -----------------

@dataclass
class ChatCollator:
    tok: Any
    max_len: int
    debug_every: int = 0  # print coverage every N batches (0=off)

    def __post_init__(self):
        if not getattr(self.tok, "is_fast", False):
            raise RuntimeError("Fast tokenizer required (offset_mapping). Reload with use_fast=True.")
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self._seen = 0

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts, msgs_per_example = [], []

        for ex in batch:
            msgs = ex.get("messages")
            if not isinstance(msgs, list) or not msgs:
                continue
            msgs = ensure_system_first(msgs)
            msgs = [{"role": m["role"], "content": canon(m.get("content", ""))} for m in msgs]
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            # Require at least one assistant span in the templated text
            if not find_assistant_char_spans_from_tags(text):
                continue
            texts.append(text)
            msgs_per_example.append(msgs)

        if not texts:
            raise RuntimeError("[collator] No valid samples in batch (no assistant spans after templating).")

        enc = self.tok(
            texts,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_len,
            return_offsets_mapping=True,
        )

        if "offset_mapping" not in enc:
            raise RuntimeError("[collator] tokenizer did not return offset_mapping")

        labels = []
        B = enc["input_ids"].size(0)
        for i in range(B):
            single = {
                "input_ids":      enc["input_ids"][i:i+1],
                "attention_mask": enc["attention_mask"][i:i+1],
                "offset_mapping": enc["offset_mapping"][i:i+1],
            }
            lab = build_labels_for_assistant(texts[i], single)
            labels.append(lab[0])

        enc["labels"] = torch.stack(labels, dim=0).to(dtype=torch.long)
        enc["input_ids"] = enc["input_ids"].to(dtype=torch.long)
        enc["attention_mask"] = enc["attention_mask"].to(dtype=torch.long)

        if self.debug_every:
            self._seen += 1
            if self._seen % self.debug_every == 0:
                total = enc["labels"].numel()
                used = int((enc["labels"] != -100).sum().item())
                pct = 100.0 * used / total if total else 0.0
                print(f"[collator] label coverage: {used}/{total} tokens = {pct:.2f}%")

        return enc


# ----------------- Safe Trainer -----------------
import torch.nn.functional as F

class SafeTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dbg = 0

    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        if isinstance(inputs, dict):
            for k in ("input_ids", "attention_mask", "labels"):
                if k in inputs and inputs[k] is not None:
                    inputs[k] = inputs[k].long()
        return inputs

    # accept extra kwargs like num_items_in_batch
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            labels=inputs.get("labels"),
        )
        loss = getattr(outputs, "loss", None)

        if loss is None or (isinstance(loss, torch.Tensor) and not torch.isfinite(loss).item()):
            logits = outputs.logits  # [B, T, V]
            labels = inputs["labels"]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if self.state.global_step < 3 and self._dbg < 3:
            with torch.no_grad():
                labels = inputs["labels"]
                total = labels.numel()
                used = int((labels != -100).sum().item())
                pct = 100.0 * used / total if total else 0.0
                try:
                    logits_std = float(outputs.logits.float().std().item())
                except Exception:
                    logits_std = float("nan")
                print(f"[debug] step={self.state.global_step} "
                      f"label_coverage={used}/{total} ({pct:.2f}%) "
                      f"logits_std={logits_std:.4f} loss={float(loss.item()):.4f}",
                      flush=True)
                self._dbg += 1

        return (loss, outputs) if return_outputs else loss


# ----------------- Main -----------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--train", type=str, required=True)
    p.add_argument("--val", type=str, required=True)
    p.add_argument("--out", type=str, required=True)

    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--mbatch", type=int, default=1)
    p.add_argument("--accum", type=int, default=16)
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)

    # LoRA
    p.add_argument("--r", type=int, default=32)
    p.add_argument("--alpha", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.05)

    # 4-bit / QLoRA
    p.add_argument("--no-4bit", action="store_true")

    # Misc
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--adam8bit", action="store_true")
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--log_steps", type=int, default=20)
    p.add_argument("--debug_labels", type=int, default=0, help="Print label coverage every N batches (0=off)")
    p.add_argument("--peek", type=int, default=0, help="If >0, print a templated sample + spans and exit.")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    if args.peek > 0:
        ds = JsonlChatDataset(args.train)
        ex = ds[0]
        msgs = ensure_system_first(ex["messages"])
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        print("=== PEEK: templated (first 600) ===")
        print(text[:600].replace("\n","\\n"))
        spans = find_assistant_char_spans_from_tags(text)
        print("=== PEEK: spans ===", spans[:10])
        enc = tok([text], padding=True, return_tensors="pt", truncation=True, max_length=args.seq, return_offsets_mapping=True)
        lab = build_labels_for_assistant(text, enc)[0]
        cov = (lab != -100).sum().item() / lab.numel() * 100
        print(f"=== PEEK: label coverage = {cov:.2f}% on first sample ===")
        return

    load_kwargs = dict(trust_remote_code=True, device_map="auto")
    if not args.no_4bit and HAS_BNB:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(args.base, **load_kwargs)
    model.config.use_cache = False  # ensure compatibility with grad checkpointing/Trainer

    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None and hasattr(gen_cfg, "sliding_window") and gen_cfg.sliding_window:
        gen_cfg.sliding_window = None

    if not args.no_4bit and HAS_BNB:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.grad_ckpt)
    elif args.grad_ckpt:
        model.gradient_checkpointing_enable()

    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "W_pack", "wi", "wo"
    ]
    lora_cfg = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    train_ds = JsonlChatDataset(args.train)
    val_ds   = JsonlChatDataset(args.val)
    collator = ChatCollator(tok=tok, max_len=args.seq, debug_every=args.debug_labels)

    optim = "paged_adamw_8bit" if args.adam8bit and HAS_BNB else "adamw_torch"

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16_ok = torch.cuda.is_available()
    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.mbatch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        logging_steps=args.log_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        gradient_checkpointing=args.grad_ckpt,
        bf16=bf16_ok,
        fp16=(not bf16_ok and fp16_ok),
        optim=optim,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = SafeTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tok,
    )

    trainer.train()
    trainer.save_model()
    tok.save_pretrained(args.out)

    print("\n=== Training complete ===")
    print(f"Saved to: {args.out}")
    print(f"Base: {args.base}")
    print(f"4-bit: {not args.no_4bit and HAS_BNB}")
    print(f"LoRA: r={args.r}, alpha={args.alpha}, dropout={args.dropout}")
    print("=====================================\n")


if __name__ == "__main__":
    main()
