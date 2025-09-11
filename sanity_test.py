#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, pathlib, re, json, sys
import torch

# ---------- Repo root + .env ----------
REPO_ROOT = pathlib.Path(__file__).resolve().parent

def expand_rel(p: str) -> str:
    if not p: return p
    # absolute posix or windows drive
    if p.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", p):
        return str(pathlib.Path(p).expanduser().resolve())
    return str((REPO_ROOT / p).expanduser().resolve())

def load_env():
    try:
        from dotenv import load_dotenv
        # try CWD, then repo root
        loaded = False
        for cand in [pathlib.Path.cwd() / ".env", REPO_ROOT / ".env"]:
            if cand.exists():
                load_dotenv(dotenv_path=cand, override=False)
                print(f"[env] loaded: {cand}")
                loaded = True
                break
        if not loaded:
            load_dotenv(override=False)  # default walk-up
            print("[env] loaded via default search")
    except Exception:
        print("[env] python-dotenv not installed; relying on process env")

load_env()

import os
from transformers import StoppingCriteria, StoppingCriteriaList

# ---- env / flags with safe defaults ----
DO_SAMPLE = os.getenv("DO_SAMPLE", "0") not in ("0", "false", "False", "")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
MAX_NEW = int(os.getenv("MAX_NEW", "80"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.05"))

print(f"[gen] max_new={MAX_NEW}  temperature={TEMPERATURE}  top_p={TOP_P}  do_sample={DO_SAMPLE}")

# ---------- CLI ----------
ap = argparse.ArgumentParser(description="Minimal sanity test for base+LoRA")
ap.add_argument("--base", default=os.getenv("MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct"),
                help="HF base model id (default from .env MODEL_ID)")
ap.add_argument("--lora", default=os.getenv("LORA_ADAPTER_DIR", "./ckpts/toddric-llama-8B-lora"),
                help="Path to LoRA adapter dir (default from .env LORA_ADAPTER_DIR)")
ap.add_argument("--no-lora", action="store_true", help="Ignore adapter; run base only")
ap.add_argument("--system-file", default=None, help="Optional system prompt file")
ap.add_argument("--prompt-file", default=None, help="Optional user prompt file")
ap.add_argument("--system", default=None, help="System text (overrides --system-file)")
ap.add_argument("--prompt", default="What is your name? Answer in 1 short line.",
                help="User text (overrides --prompt-file)")
ap.add_argument("--max-new", type=int, default=220)
ap.add_argument("--temperature", type=float, default=0.7)
ap.add_argument("--top-p", type=float, default=0.95)
ap.add_argument("--load-in-4bit", action="store_true", help="Load base in 4-bit (recommended on 16GB VRAM)")
ap.add_argument("--device", default="auto", choices=["auto","cuda","cpu"], help="Force device (default auto)")
args = ap.parse_args()

BASE_ID = args.base
LORA_DIR = expand_rel(args.lora) if args.lora else None

print("────────────────────────────────────────────────────────")
print(f"[sanity] Repo root     : {REPO_ROOT}")
print(f"[sanity] Base model    : {BASE_ID}")
print(f"[sanity] LoRA adapter  : {LORA_DIR if not args.no_lora else '(disabled via --no-lora)'}")
print(f"[sanity] 4-bit load    : {args.load_in_4bit}")
print(f"[sanity] Device        : {args.device}")
print(f"[gen] max_new={args.max_new}  temperature={args.temperature}  top_p={args.top_p}")
print("────────────────────────────────────────────────────────")

# ---------- Load model ----------
from transformers import AutoTokenizer, AutoModelForCausalLM

def pick_dtype():
    if torch.cuda.is_available():
        major = torch.cuda.get_device_capability(0)[0]
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32

quant_cfg = None
if args.load_in_4bit:
    try:
        from transformers import BitsAndBytesConfig
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=pick_dtype(),
            bnb_4bit_use_double_quant=True
        )
    except Exception as e:
        print(f"[warn] bitsandbytes not available for 4-bit: {e}. Falling back to full precision.")
        quant_cfg = None

tok = AutoTokenizer.from_pretrained(BASE_ID, use_fast=True)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_ID,
    torch_dtype=pick_dtype(),
    device_map="auto" if args.device=="auto" else None,
    attn_implementation="eager",
    quantization_config=quant_cfg
)

# Move to explicit device if requested
if args.device != "auto":
    device = torch.device("cuda" if args.device=="cuda" and torch.cuda.is_available() else "cpu")
    model.to(device)

# Apply LoRA adapter
if (not args.no_lora) and LORA_DIR:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, LORA_DIR)
    model.eval()
    print(f"[sanity] Applied LoRA from {LORA_DIR}")

# ---------- Build messages ----------
def file_text(path):
    if not path: return None
    p = pathlib.Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else None

system_text = args.system or file_text(args.system_file) or "You are Toddric — pragmatic, nerdy, playful, and wise."
prompt_text = args.prompt or file_text(args.prompt_file) or "Hello!"

messages = [
    {"role": "system", "content": system_text},
    {"role": "user", "content": prompt_text},
]

text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tok(text, return_tensors="pt")
if args.device != "cpu" and torch.cuda.is_available():
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

gen_kwargs = dict(
    max_new_tokens=args.max_new,
    do_sample=True if args.temperature > 0 else False,
    temperature=args.temperature,
    top_p=args.top_p,
    pad_token_id=tok.pad_token_id,
    eos_token_id=tok.eos_token_id,
)

with torch.no_grad():
    out = model.generate(**inputs, **gen_kwargs)

decoded = tok.decode(out[0], skip_special_tokens=True)

# Try to print only the assistant portion after the prompt
try:
    # Find the last occurrence of the user content and print what follows
    idx = decoded.rfind(prompt_text.strip())
    reply = decoded[idx + len(prompt_text.strip()):].strip() if idx != -1 else decoded
    print("\n=== Assistant ===")
    print(reply)
except Exception:
    print(decoded)

