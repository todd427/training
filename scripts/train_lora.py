#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LoRA / QLoRA training with .env-driven configuration (robust)
- Loads .env before torch/transformers (ENV_FILE, CWD/.env, repo/.env, parent/.env)
- EnvDefaults picks up MODEL_ID, TRAIN_JSONL, VAL_JSONL, OUT_DIR (or LORA_ADAPTER_DIR)
- Chat JSONL dataset with assistant-only loss masking (fast tokenizer required)
- dtype API: try dtype= then fallback to torch_dtype= (no deprecation spam)
- Attention auto-picker: FlashAttention-2 -> SDPA
- device_map=auto safe: disable Trainer's device move if needed; optional VRAM caps + CPU offload
- QLoRA turns ON only if bitsandbytes is importable; otherwise cleanly OFF
- Step-wise eval + save-best (by eval_loss), prints eval PPL
- Optional post-train sampling log
"""

# ── Load .env BEFORE importing torch / transformers ───────────────────────────
import os, re, json, math, inspect, argparse, pathlib
from dataclasses import dataclass, asdict
from typing import Dict, List

def _load_env_early():
    try:
        from dotenv import load_dotenv
    except Exception:
        print("[env] python-dotenv not installed; using process env only.")
        return
    env_file = os.getenv("ENV_FILE")
    candidates = []
    if env_file:
        candidates.append(pathlib.Path(env_file))
    here = pathlib.Path(__file__).resolve().parent
    candidates += [
        pathlib.Path.cwd() / ".env",
        here.parent / ".env",
        here.parent.parent / ".env",
    ]
    for p in candidates:
        try:
            if p and p.exists():
                load_dotenv(p, override=False)
                print(f"[env] loaded: {p}")
                return
        except Exception:
            pass
    try:
        load_dotenv(override=False)
        print("[env] loaded via default search")
    except Exception:
        print("[env] no .env file loaded")

_load_env_early()

# ── Paths / repo root helper ──────────────────────────────────────────────────
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
print("[env] REPO_ROOT:", REPO_ROOT)

def expand_rel(p: str) -> str:
    if not p:
        return p
    if p.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", p):
        return str(pathlib.Path(p).expanduser().resolve())
    return str((REPO_ROOT / p).expanduser().resolve())

# ── Quantization deps detection (robust) ──────────────────────────────────────
_HAS_BNB = False
try:
    import bitsandbytes as bnb  # verifies the actual package is importable
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

# Prefer the class exported by transformers; fall back to internal bnb path.
try:
    from transformers import BitsAndBytesConfig
except Exception:
    try:
        from bitsandbytes.config import BitsAndBytesConfig  # may not exist in some installs
    except Exception:
        BitsAndBytesConfig = None

# ── Core deps ─────────────────────────────────────────────────────────────────
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    Trainer, TrainingArguments, set_seed, trainer as _trainer_mod
)

# ── Accelerate unwrap_model shim (compat) ─────────────────────────────────────
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
def pick_dtype() -> torch.dtype:
    """Prefer bfloat16 on Hopper/Ada; else fp16 on older CUDA; else fp32 CPU."""
    if torch.cuda.is_available():
        maj = torch.cuda.get_device_capability(0)[0]
        return torch.bfloat16 if maj >= 8 else torch.float16
    return torch.float32

def canon(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def _flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False

def pick_attn_impl() -> str:
    return "flash_attention_2" if _flash_attn_available() else "sdpa"

# ── Dataset / Collator (assistant-only loss) ──────────────────────────────────
class JsonlDataset(Dataset):
    """Each line: {"messages":[{"role":"system","content":"..."}, ... ]}"""
    def __init__(self, path: str):
        self.rows: List[dict] = []
        path = os.path.expanduser(path)
        with open(path, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    o = json.loads(ln)
                except Exception:
                    continue
                if isinstance(o, dict) and isinstance(o.get("messages"), list) and len(o["messages"]) >= 2:
                    self.rows.append(o)
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

def build_labels_for_assistant(tok, text: str, messages: List[dict], enc) -> torch.Tensor:
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
    def __init__(self, tok, max_len: int):
        self.tok = tok; self.max_len = max_len
        if not getattr(tok, "is_fast", False):
            raise RuntimeError("Fast tokenizer required (return_offsets_mapping).")
    def __call__(self, batch: List[dict]):
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

# ── TrainingArguments helper (API-compat) ─────────────────────────────────────
def make_training_args(eval_steps: int, save_steps: int, **kw) -> TrainingArguments:
    TA = TrainingArguments
    params = set(inspect.signature(TA.__init__).parameters.keys())

    # evaluation strategy
    if "evaluation_strategy" in params:
        kw["evaluation_strategy"] = "steps"
    elif "eval_strategy" in params:
        kw["eval_strategy"] = "steps"

    # save strategy + intervals
    if "save_strategy" in params:
        kw["save_strategy"] = "steps"
    if "eval_steps" in params:
        kw["eval_steps"] = eval_steps
    if "save_steps" in params:
        kw["save_steps"] = save_steps

    # best checkpoint recovery
    if "load_best_model_at_end" in params:
        kw["load_best_model_at_end"] = True
    if "metric_for_best_model" in params:
        kw["metric_for_best_model"] = "eval_loss"
    if "greater_is_better" in params:
        kw["greater_is_better"] = False

    # Avoid Trainer re-moving a sharded model when using device_map=auto
    if "place_model_on_device" in params:
        kw.setdefault("place_model_on_device", False)

    # QoL defaults
    if "tf32" in params:
        kw.setdefault("tf32", True)
    if "logging_steps" in params:
        kw.setdefault("logging_steps", max(1, eval_steps // 2))

    # Prefer 8-bit optimizer states when bitsandbytes is available
    if "optim" in params and _HAS_BNB:
        kw.setdefault("optim", "adamw_bnb_8bit")

    supported = {k: v for k, v in kw.items() if k in params}
    return TA(**supported)

# ── Trainer compat: disable auto-move when needed ─────────────────────────────
def _disable_trainer_move_if_needed(device_map: str):
    """If TrainingArguments doesn't support `place_model_on_device` and we're
    using a sharded device_map, neutralize Trainer's mover to avoid meta crash."""
    try:
        ta_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    except Exception:
        ta_params = set()
    sharded = (device_map not in (None, "", "cpu", "cuda:0"))
    if "place_model_on_device" not in ta_params and sharded:
        _orig = _trainer_mod.Trainer._move_model_to_device
        def _no_move(self, model, device):
            return model
        _trainer_mod.Trainer._move_model_to_device = _no_move
        print(f"[compat] Patched Trainer._move_model_to_device (device_map={device_map})")

# ── Config dataclass from .env ────────────────────────────────────────────────
@dataclass
class EnvDefaults:
    # core paths
    base: str = os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
    train: str = os.getenv("TRAIN_JSONL", "")
    val:   str = os.getenv("VAL_JSONL", "")
    out:   str = os.getenv("OUT_DIR", "") or os.getenv("LORA_ADAPTER_DIR", "")  # alias

    # quant / device
    no_4bit: bool = os.getenv("NO_4BIT", "0") in ("1","true","True")
    load_in_4bit: bool = os.getenv("LOAD_IN_4BIT", "0") in ("1","true","True")  # informational
    device_map_env: str = os.getenv("DEVICE_MAP", "auto")
    torch_dtype_env: str = os.getenv("TORCH_DTYPE", "auto")
    torch_device: str = os.getenv("TORCH_DEVICE", "").strip()  # e.g., "cuda:0"

    # training core
    seq: int = int(os.getenv("SEQ_LEN", "2048"))
    epochs: int = int(os.getenv("EPOCHS", "2"))
    mbatch: int = int(os.getenv("MBATCH", "1"))
    accum: int = int(os.getenv("ACCUM", "16"))
    eval_steps: int = int(os.getenv("EVAL_STEPS", "250"))
    save_steps: int = int(os.getenv("SAVE_STEPS", "500"))
    r: int = int(os.getenv("LORA_R", "32"))
    alpha: int = int(os.getenv("LORA_ALPHA", "64"))
    dropout: float = float(os.getenv("LORA_DROPOUT", "0.05"))
    num_workers: int = int(os.getenv("NUM_WORKERS", "0"))
    lr: float = float(os.getenv("LR", "2e-4"))
    wd: float = float(os.getenv("WEIGHT_DECAY", "0.0"))
    warmup_ratio: float = float(os.getenv("WARMUP_RATIO", "0.05"))
    seed: int = int(os.getenv("SEED", "1337"))

    # sampling/logging after training
    do_sample: bool = os.getenv("DO_SAMPLE", "1") in ("1","true","True")
    temperature: float = float(os.getenv("TEMPERATURE", "0.3"))
    top_p: float = float(os.getenv("TOP_P", "0.9"))
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "256"))
    repetition_penalty: float = float(os.getenv("REPETITION_PENALTY", "1.12"))
    log_transcripts: bool = os.getenv("LOG_TRANSCRIPTS", "0") in ("1","true","True")
    log_dir: str = os.getenv("LOG_DIR", "./logs")

    def __post_init__(self):
        if not self.train: raise RuntimeError("TRAIN_JSONL not set")
        if not self.val:   raise RuntimeError("VAL_JSONL not set")
        if not self.out:   raise RuntimeError("OUT_DIR (or LORA_ADAPTER_DIR) not set")
        self.train = expand_rel(self.train)
        self.val   = expand_rel(self.val)
        self.out   = expand_rel(self.out)

# ── Model / LoRA helpers ──────────────────────────────────────────────────────
def _parse_dtype_label(lbl: str, auto: torch.dtype) -> torch.dtype:
    s = (lbl or "auto").lower()
    return {
        "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
        "fp16": torch.float16, "half": torch.float16,
        "fp32": torch.float32, "float32": torch.float32,
    }.get(s, auto)

def load_base(model_id: str, use_4bit: bool, device_map: str, forced_dtype: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    cfg = AutoConfig.from_pretrained(model_id)
    if getattr(cfg, "sliding_window", None):
        cfg.sliding_window = None

    quant_cfg = None
    if use_4bit and _HAS_BNB and BitsAndBytesConfig is not None:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=pick_dtype(),
            bnb_4bit_use_double_quant=True,
        )

    want_dtype = _parse_dtype_label(forced_dtype, pick_dtype())
    attn_impl = pick_attn_impl()

    common = dict(
        device_map=device_map,
        attn_implementation=attn_impl,
        quantization_config=quant_cfg,
        low_cpu_mem_usage=True,
    )
    # Cap VRAM and offload when auto-sharding to avoid OOM spikes
    if device_map == "auto" and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory // (1024**2)  # MiB
        cap = max(12000, int(total * 0.90))  # leave headroom
        common["max_memory"] = {0: f"{cap}MiB", "cpu": "64GiB"}
        off_dir = os.path.join(os.environ.get("OFFLOAD_DIR", "."), "offload")
        os.makedirs(off_dir, exist_ok=True)
        common["offload_folder"] = off_dir
        print(f"[load] max_memory cap={cap}MiB, offload_folder={off_dir}")

    # Try new API first (dtype=), then gracefully fall back to torch_dtype=
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=want_dtype, **common)
    except TypeError as e:
        if "dtype" in str(e):
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=want_dtype, **common)
        else:
            raise

    if getattr(model.generation_config, "sliding_window", None):
        model.generation_config.sliding_window = None
    model.config.use_cache = False

    if quant_cfg:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

    # Leaner checkpointing to reduce spikes
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()

    return tok, model

def build_peft(model, r=32, alpha=64, dropout=0.05, target_modules=None):
    from peft import LoraConfig, get_peft_model
    if target_modules is None:
        target_modules = ["q_proj","k_proj","v_proj","o_proj"]
    lcfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    return get_peft_model(model, lcfg)

# ── Post-train sampling / logging ─────────────────────────────────────────────
def sample_and_log(model, tok, envd: EnvDefaults):
    if not envd.log_transcripts:
        return
    os.makedirs(envd.log_dir, exist_ok=True)
    path = os.path.join(envd.log_dir, "train_lora_samples.txt")
    prompts = [
        "You are Toddric. Summarize briefly why LoRA adapters are useful.",
        "Offer a two-sentence tip for authors debugging a Hugging Face dataset.",
    ]
    model.eval()
    with open(path, "a", encoding="utf-8") as f, torch.inference_mode():
        for p in prompts:
            ids = tok.apply_chat_template(
                [{"role":"system","content":"You are a helpful assistant."},
                 {"role":"user","content":p}],
                add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)
            out = model.generate(
                ids,
                do_sample=envd.do_sample,
                temperature=envd.temperature,
                top_p=envd.top_p,
                max_new_tokens=envd.max_new_tokens,
                repetition_penalty=envd.repetition_penalty,
            )
            text = tok.decode(out[0], skip_special_tokens=True)
            f.write("\n\n### PROMPT\n" + p + "\n### OUTPUT\n" + text + "\n")
    print(f"[log] samples → {path}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    envd = EnvDefaults()

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=envd.base)
    ap.add_argument("--train", default=envd.train)
    ap.add_argument("--val",   default=envd.val)
    ap.add_argument("--out",   default=envd.out)
    ap.add_argument("--seq", type=int, default=envd.seq)
    ap.add_argument("--epochs", type=int, default=envd.epochs)
    ap.add_argument("--mbatch", type=int, default=envd.mbatch)
    ap.add_argument("--accum", type=int, default=envd.accum)
    ap.add_argument("--eval_steps", type=int, default=envd.eval_steps)
    ap.add_argument("--save_steps", type=int, default=envd.save_steps)
    ap.add_argument("--r", type=int, default=envd.r)
    ap.add_argument("--alpha", type=int, default=envd.alpha)
    ap.add_argument("--dropout", type=float, default=envd.dropout)
    ap.add_argument("--num_workers", type=int, default=envd.num_workers)
    ap.add_argument("--lr", type=float, default=envd.lr)
    ap.add_argument("--wd", type=float, default=envd.wd)
    ap.add_argument("--warmup_ratio", type=float, default=envd.warmup_ratio)
    ap.add_argument("--seed", type=int, default=envd.seed)
    ap.add_argument("--no-4bit", action="store_true", default=envd.no_4bit)
    ap.add_argument("--device_map", default=envd.device_map_env)
    ap.add_argument("--torch_dtype", default=envd.torch_dtype_env)
    ap.add_argument("--torch_device", default=envd.torch_device)
    # logging/sampling
    ap.add_argument("--log_transcripts", action="store_true", default=envd.log_transcripts)
    ap.add_argument("--log_dir", default=envd.log_dir)
    ap.add_argument("--do_sample", action="store_true", default=envd.do_sample)
    ap.add_argument("--temperature", type=float, default=envd.temperature)
    ap.add_argument("--top_p", type=float, default=envd.top_p)
    ap.add_argument("--max_new_tokens", type=int, default=envd.max_new_tokens)
    ap.add_argument("--repetition_penalty", type=float, default=envd.repetition_penalty)
    args = ap.parse_args()

    print(f"[train_lora] Base model: {args.base}")
    print(f"[train_lora] Training  : {args.train}")
    print(f"[train_lora] Validation : {args.val}")
    print(f"[train_lora] Output dir : {args.out}")
    print(f"[train_lora] Device map : {args.device_map}")
    # Respect only cuda:N forms for TORCH_DEVICE
    if args.torch_device:
        m = re.fullmatch(r"cuda:(\d+)", str(args.torch_device).strip())
        if m:
            try:
                torch.cuda.set_device(int(m.group(1)))
                print(f"[env] TORCH_DEVICE -> {args.torch_device}")
            except Exception as e:
                print(f"[warn] TORCH_DEVICE ignored: {e}")
    print("──────────────────────────────────────────────")

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # QLoRA ON if bitsandbytes is importable and NO_4BIT is not set
    use_4bit = _HAS_BNB and (not args.no_4bit)
    print(f"[train_lora] QLoRA 4-bit: {'ON' if use_4bit else 'OFF'}")

    # Load base + tokenizer
    tok, base = load_base(args.base, use_4bit=use_4bit, device_map=args.device_map, forced_dtype=args.torch_dtype)

    # Disable Trainer move if TrainingArguments lacks flag and we shard
    _disable_trainer_move_if_needed(args.device_map)

    # LoRA wrap
    model = build_peft(base, r=args.r, alpha=args.alpha, dropout=args.dropout)

    # Datasets + collator
    train_ds = JsonlDataset(args.train)
    val_ds   = JsonlDataset(args.val)
    collate  = ChatCollator(tok, args.seq)

    steps_per_epoch = math.ceil(len(train_ds) / max(1, (args.mbatch * args.accum)))
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
        bf16=(pick_dtype()==torch.bfloat16),
        fp16=(pick_dtype()==torch.float16),
        dataloader_pin_memory=True,
        dataloader_num_workers=args.num_workers,
        gradient_checkpointing=True,
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
        save_total_limit=2,
    )
    targs = make_training_args(args.eval_steps, args.save_steps, **common)

    # Tokenizer vs processing_class compatibility
    kw = dict(model=model, args=targs, train_dataset=train_ds,
              eval_dataset=val_ds, data_collator=collate)
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        kw["processing_class"] = tok
    else:
        kw["tokenizer"] = tok

    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.out, "config_used.json"), "w") as w:
        json.dump({
            "repo_root": str(REPO_ROOT),
            "env_defaults": asdict(EnvDefaults()),  # snapshot of env-derived defaults
            "resolved_args": vars(args),
            "dtype": str(pick_dtype()),
            "use_4bit": use_4bit,
        }, w, indent=2)

    trainer = Trainer(**kw)
    trainer.train()

    # Evaluate best
    metrics = trainer.evaluate()
    if "eval_loss" in metrics:
        ppl = math.exp(min(20.0, metrics["eval_loss"]))
        print(f"[eval] eval_loss={metrics['eval_loss']:.4f} | ppl≈{ppl:.2f}")
    else:
        print("[eval] no eval_loss in metrics; check eval config")

    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"[done] LoRA saved → {args.out}")

    # Optional sampling log
    if args.log_transcripts:
        sample_and_log(model, tok, EnvDefaults())

if __name__ == "__main__":
    main()

