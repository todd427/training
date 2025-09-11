#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA / QLoRA training driven by .env

- Loads .env from CWD, repo root (parent of scripts/), or ENV_FILE
- Requires TRAIN_JSONL, VAL_JSONL, OUT_DIR after .env is loaded
- Resolves relative paths against repo root
- Assistant-only loss collator (fast tokenizer required)
- Supports 4-bit QLoRA with bitsandbytes
"""

import os, re, json, math, inspect, argparse, pathlib, pprint
from dataclasses import dataclass, asdict
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    Trainer, TrainingArguments
)

# ── Paths / repo root ─────────────────────────────────────────────────────────
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
# repo root is the parent of scripts/
REPO_ROOT = SCRIPT_DIR.parent
print("REPO_ROOT :", REPO_ROOT)

def expand_rel(p: str) -> str:
    if not p:
        return p
    # Treat anything not starting with "/" (POSIX) or a drive letter as relative
    if p.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", p):
        return str(pathlib.Path(p).expanduser().resolve())
    return str((REPO_ROOT / p).expanduser().resolve())

# ── .env loader with helpful prints ───────────────────────────────────────────
def load_env_file():
    """
    Try ENV_FILE, then CWD/.env, then REPO_ROOT/.env.
    Print what was loaded so we never wonder again.
    """
    try:
        from dotenv import load_dotenv
    except Exception:
        # dotenv not installed; rely on process env
        print("[env] python-dotenv not installed; using process env only.")
        return None

    candidates = []
    env_file = os.getenv("ENV_FILE")
    print("env_file :", env_file)
    if env_file:
        candidates.append(pathlib.Path(env_file))
    candidates += [
        pathlib.Path.cwd() / ".env",
        REPO_ROOT / ".env",
        REPO_ROOT.parent / ".env",
    ]

    for p in candidates:
        try:
            if p and p.exists():
                load_dotenv(p, override=False)
                print(f"[env] loaded: {p}")
                return str(p)
        except Exception:
            pass

    # Fallback to default search (dotenv walks up from CWD)
    try:
        load_dotenv(override=False)
        print("[env] loaded via default search (no explicit file detected)")
    except Exception:
        print("[env] no .env file loaded (default search failed)")
    return None

_which_env = load_env_file()

# ── Optional 4-bit quantization ───────────────────────────────────────────────
try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    BitsAndBytesConfig = None
    _HAS_BNB = False

# ── PEFT / LoRA ───────────────────────────────────────────────────────────────
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ── Accelerate unwrap_model shim (compat across versions) ─────────────────────
try:
    from accelerate import Accelerator
    if "keep_torch_compile" not in inspect.signature(Accelerator.unwrap_model).parameters:
        _orig_unwrap = Accelerator.unwrap_model
        def _unwrap_shim(self, model, **kwargs):
            return _orig_unwrap(self, model)
        Accelerator.unwrap_model = _unwrap_shim
except Exception:
    pass

# ── Utils ─────────────────────────────────────────────────────────────────────
def pick_dtype():
    if torch.cuda.is_available():
        major = torch.cuda.get_device_capability(0)[0]
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32

def canon(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

# ── Dataset / Collator ────────────────────────────────────────────────────────
class JsonlDataset(Dataset):
    def __init__(self, path):
        self.rows = []
        path = os.path.expanduser(path)
        with open(path, encoding="utf-8") as f:
            for ln in f:
                try:
                    o = json.loads(ln)
                except Exception:
                    continue
                if isinstance(o, dict) and isinstance(o.get("messages"), list) and len(o["messages"]) >= 2:
                    self.rows.append(o)
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

def build_labels_for_assistant(tok, text, messages, enc):
    spans = []; search_from = 0
    for m in messages:
        if m.get("role") != "assistant":
            continue
        content = canon(m.get("content", ""))
        if not content:
            continue
        idx = text.find(content, search_from)
        if idx == -1:
            continue
        spans.append((idx, idx + len(content)))
        search_from = idx + len(content)
    labels = torch.full_like(enc["input_ids"], -100)
    offs = enc["offset_mapping"][0].tolist()
    for i, (a, b) in enumerate(offs):
        if a == b == 0:
            continue
        for (s, e) in spans:
            if not (b <= s or a >= e):
                labels[0, i] = enc["input_ids"][0, i]
                break
    return labels

class ChatCollator:
    def __init__(self, tok, max_len):
        self.tok = tok; self.max_len = max_len
        if not getattr(tok, "is_fast", False):
            raise RuntimeError("Fast tokenizer required (return_offsets_mapping).")
    def __call__(self, batch):
        input_ids, attn, labels = [], [], []
        for ex in batch:
            msgs = ex["messages"]
            if msgs[0].get("role") != "system":
                msgs = [{"role": "system", "content": "You are a helpful assistant."}] + msgs
            msgs = [{"role": m["role"], "content": canon(m["content"])} for m in msgs]
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            enc = self.tok(text, return_tensors="pt", truncation=True,
                           max_length=self.max_len, return_offsets_mapping=True)
            lab = build_labels_for_assistant(self.tok, text, msgs, enc)
            input_ids.append(enc["input_ids"][0])
            attn.append(enc["attention_mask"][0])
            labels.append(lab[0])
        batch_enc = self.tok.pad({"input_ids": input_ids, "attention_mask": attn},
                                 padding=True, return_tensors="pt")
        maxlen = batch_enc["input_ids"].size(1)
        padded = [torch.nn.functional.pad(lab, (maxlen - lab.size(0), 0), value=-100) for lab in labels]
        batch_enc["labels"] = torch.stack(padded, dim=0).to(dtype=torch.long)
        return batch_enc

# ── Model / LoRA helpers ──────────────────────────────────────────────────────
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
    names = set()
    for n, _ in model.named_modules():
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
    params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    if "eval_strategy" in params:
        return TrainingArguments(eval_strategy="steps", eval_steps=eval_steps, **kw)
    elif "evaluation_strategy" in params:
        return TrainingArguments(evaluation_strategy="steps", eval_steps=eval_steps, **kw)
    return TrainingArguments(**kw)

# ── Config dataclass (validated after .env load) ──────────────────────────────
@dataclass
class EnvDefaults:
    base: str = os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
    train: str = os.getenv("TRAIN_JSONL", "")
    val:   str = os.getenv("VAL_JSONL", "")
    out:   str = os.getenv("OUT_DIR", "")
    seq: int = int(os.getenv("SEQ_LEN", "2048"))
    lr: float = float(os.getenv("LR", "2e-4"))
    wd: float = float(os.getenv("WEIGHT_DECAY", "0.0"))
    epochs: int = int(os.getenv("EPOCHS", "2"))
    mbatch: int = int(os.getenv("MBATCH", "1"))
    accum: int = int(os.getenv("ACCUM", "16"))
    warmup_ratio: float = float(os.getenv("WARMUP_RATIO", "0.05"))
    save_steps: int = int(os.getenv("SAVE_STEPS", "500"))
    eval_steps: int = int(os.getenv("EVAL_STEPS", "250"))
    r: int = int(os.getenv("LORA_R", "32"))
    alpha: int = int(os.getenv("LORA_ALPHA", "64"))
    dropout: float = float(os.getenv("LORA_DROPOUT", "0.05"))
    seed: int = int(os.getenv("SEED", "1337"))
    no_4bit: bool = os.getenv("NO_4BIT", "0") in ("1","true","True")
    num_workers: int = int(os.getenv("NUM_WORKERS", "0"))

    def __post_init__(self):
        # Fail fast if required keys are missing
        if not self.train:
            raise RuntimeError("TRAIN_JSONL not set in .env (or environment)")
        if not self.val:
            raise RuntimeError("VAL_JSONL not set in .env (or environment)")
        if not self.out:
            raise RuntimeError("OUT_DIR not set in .env (or environment)")

        # Normalize relative paths against repo root
        self.train = expand_rel(self.train)
        self.val   = expand_rel(self.val)
        self.out   = expand_rel(self.out)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    envd = EnvDefaults()

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=envd.base)
    ap.add_argument("--train", default=envd.train)
    ap.add_argument("--val",   default=envd.val)
    ap.add_argument("--out",   default=envd.out)
    ap.add_argument("--seq", type=int, default=envd.seq)
    ap.add_argument("--lr", type=float, default=envd.lr)
    ap.add_argument("--wd", type=float, default=envd.wd)
    ap.add_argument("--epochs", type=int, default=envd.epochs)
    ap.add_argument("--mbatch", type=int, default=envd.mbatch)
    ap.add_argument("--accum", type=int, default=envd.accum)
    ap.add_argument("--warmup_ratio", type=float, default=envd.warmup_ratio)
    ap.add_argument("--save_steps", type=int, default=envd.save_steps)
    ap.add_argument("--eval_steps", type=int, default=envd.eval_steps)
    ap.add_argument("--r", type=int, default=envd.r)
    ap.add_argument("--alpha", type=int, default=envd.alpha)
    ap.add_argument("--dropout", type=float, default=envd.dropout)
    ap.add_argument("--seed", type=int, default=envd.seed)
    ap.add_argument("--no-4bit", action="store_true", default=envd.no_4bit)
    ap.add_argument("--num-workers", type=int, default=envd.num_workers)
    args = ap.parse_args()

    print(f"[train_lora] .env file: {(_which_env or 'default search / process env')}")
    print(f"[train_lora] Repo root : {REPO_ROOT}")
    print(f"[train_lora] Base model: {args.base}")
    print(f"[train_lora] Training data: {args.train}")
    print(f"[train_lora] Validation data: {args.val}")
    print(f"[train_lora] LoRA output: {args.out}")
    print("──────────────────────────────────────────────")

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

    kw = dict(model=model, args=targs, train_dataset=train_ds,
              eval_dataset=val_ds, data_collator=collate)
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        kw["processing_class"] = tok
    else:
        kw["tokenizer"] = tok

    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.out, "config_used.json"), "w") as w:
        json.dump({
            "env_file": _which_env,
            "repo_root": str(REPO_ROOT),
            "env_defaults": asdict(envd),
            "resolved_args": vars(args),
            "dtype": str(pick_dtype()),
            "use_4bit": use_4bit,
        }, w, indent=2)

    trainer = Trainer(**kw)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"[done] LoRA saved → {args.out}")

if __name__ == "__main__":
    main()

