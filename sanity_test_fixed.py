#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sanity_test.py — deterministic sanity check for base vs LoRA with strict param handling.

- Loads .env (cwd or script dir).
- CLI overrides .env; .env overrides hard defaults.
- Optional stop sentinel.
- Optional post-processing to enforce EXACTLY N bullets.
"""

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
    except Exception as e:
        print(f"[env] python-dotenv not installed or failed ({e}); relying on process env")

load_env()

# ---------- CLI ----------
ap = argparse.ArgumentParser(description="Minimal sanity test for base+LoRA")
ap.add_argument("--base", default=None, help="HF base model id (env: MODEL_ID)")
ap.add_argument("--lora", default=None, help="Path to LoRA adapter dir (env: LORA_ADAPTER_DIR)")
ap.add_argument("--no-lora", action="store_true", help="Ignore adapter; run base only")
ap.add_argument("--system-file", default=None, help="Optional system prompt file")
ap.add_argument("--prompt-file", default=None, help="Optional user prompt file")
ap.add_argument("--system", default=None, help="System text (overrides --system-file)")
ap.add_argument("--prompt", default=None, help="User text (overrides --prompt-file)")

ap.add_argument("--max-new", type=int, default=None, help="env: MAX_NEW (default 80)")
ap.add_argument("--temperature", type=float, default=None, help="env: TEMPERATURE (default 0.2)")
ap.add_argument("--top-p", type=float, default=None, help="env: TOP_P (default 0.9)")
ap.add_argument("--repetition-penalty", type=float, default=None, help="env: REPETITION_PENALTY (default 1.05)")
ap.add_argument("--do-sample", type=int, choices=[0,1], default=None, help="env: DO_SAMPLE (0/1). If None, inferred from temperature>0")
ap.add_argument("--stop", default=None, help='Stop sentinel string (env: STOP_SENTINEL). Example: "END"')
ap.add_argument("--enforce-bullets", type=int, choices=[0,1], default=None, help="env: ENFORCE_BULLETS (default 0)")
ap.add_argument("--bullets", type=int, default=None, help="env: BULLETS (default 3). Only used if ENFORCE_BULLETS=1")

ap.add_argument("--load-in-4bit", action="store_true", help="Load base in 4-bit (recommended on 16GB VRAM)")
ap.add_argument("--device", default=None, choices=["auto","cuda","cpu"], help="Force device (env: DEVICE, default auto)")
args = ap.parse_args()

# ---------- Resolve params: CLI > ENV > default ----------
def env_bool(key, default):
    v = os.getenv(key)
    if v is None: return default
    return not (str(v) in ("0","false","False","no",""))

BASE_ID = args.base or os.getenv("MODEL_ID") or "meta-llama/Llama-3.1-8B-Instruct"
LORA_DIR = expand_rel(args.lora or os.getenv("LORA_ADAPTER_DIR") or "./ckpts/toddric-llama-8B-lora")
DEVICE = args.device or os.getenv("DEVICE") or "auto"

MAX_NEW = args.max_new if args.max_new is not None else int(os.getenv("MAX_NEW", "80"))
TEMPERATURE = args.temperature if args.temperature is not None else float(os.getenv("TEMPERATURE", "0.2"))
TOP_P = args.top_p if args.top_p is not None else float(os.getenv("TOP_P", "0.9"))
REP_P = args.repetition_penalty if args.repetition_penalty is not None else float(os.getenv("REPETITION_PENALTY", "1.05"))
STOP_SENTINEL = args.stop if args.stop is not None else os.getenv("STOP_SENTINEL", None)

# do_sample precedence
if args.do_sample is not None:
    DO_SAMPLE = bool(args.do_sample)
elif os.getenv("DO_SAMPLE") is not None:
    DO_SAMPLE = env_bool("DO_SAMPLE", False)
else:
    DO_SAMPLE = TEMPERATURE > 0

ENFORCE_BULLETS = bool(args.enforce_bullets) if args.enforce_bullets is not None else env_bool("ENFORCE_BULLETS", False)
BULLETS = args.bullets if args.bullets is not None else int(os.getenv("BULLETS", "3"))

# ---------- Print config ----------
print("────────────────────────────────────────────────────────")
print(f"[sanity] Repo root     : {REPO_ROOT}")
print(f"[sanity] Base model    : {BASE_ID}")
print(f"[sanity] LoRA adapter  : {LORA_DIR if not args.no_lora else '(disabled via --no-lora)'}")
print(f"[sanity] 4-bit load    : {args.load_in_4bit}")
print(f"[sanity] Device        : {DEVICE}")
print(f"[gen] max_new={MAX_NEW}  temperature={TEMPERATURE}  top_p={TOP_P}  do_sample={DO_SAMPLE}  rep_pen={REP_P}")
if STOP_SENTINEL: print(f"[gen] stop sentinel    : {STOP_SENTINEL!r}")
if ENFORCE_BULLETS: print(f"[post] enforce bullets : {BULLETS}")
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
    device_map="auto" if DEVICE=="auto" else None,
    attn_implementation="eager",
    quantization_config=quant_cfg
)

# Move to explicit device if requested
if DEVICE != "auto":
    device = torch.device("cuda" if DEVICE=="cuda" and torch.cuda.is_available() else "cpu")
    model.to(device)

# Apply LoRA adapter
if (not args.no_lora) and LORA_DIR:
    try:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, LORA_DIR)
        model.eval()
        print(f"[sanity] Applied LoRA from {LORA_DIR}")
    except Exception as e:
        print(f"[warn] Failed to load LoRA from {LORA_DIR}: {e}")

# ---------- Build messages ----------
def file_text(path):
    if not path: return None
    p = pathlib.Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else None

system_text = (args.system or file_text(args.system_file)
               or os.getenv("SYSTEM_TEXT")
               or "You are a concise assistant. Follow formatting instructions exactly.")
prompt_text = (args.prompt or file_text(args.prompt_file)
               or os.getenv("PROMPT_TEXT")
               or "Say hello in one line.")

messages = [
    {"role": "system", "content": system_text},
    {"role": "user", "content": prompt_text},
]

text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tok(text, return_tensors="pt")
if DEVICE != "cpu" and torch.cuda.is_available():
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

# ---------- Stop criteria ----------
from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnString(StoppingCriteria):
    def __init__(self, tokenizer, stop_str="END"):
        self.stop_ids = tokenizer(stop_str, add_special_tokens=False).input_ids
    def __call__(self, input_ids, scores, **kwargs):
        ids = input_ids[0].tolist()
        n = len(self.stop_ids)
        return len(ids) >= n and ids[-n:] == self.stop_ids

stops = StoppingCriteriaList([])
if STOP_SENTINEL:
    stops.append(StopOnString(tok, STOP_SENTINEL))

# ---------- Generate ----------
gen_kwargs = dict(
    max_new_tokens=MAX_NEW,
    do_sample=DO_SAMPLE,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    repetition_penalty=REP_P,
    pad_token_id=tok.pad_token_id,
    eos_token_id=tok.eos_token_id,
    stopping_criteria=stops if len(stops)>0 else None,
)

with torch.no_grad():
    out = model.generate(**inputs, **gen_kwargs)

decoded = tok.decode(out[0], skip_special_tokens=True)

# ---------- Extract the assistant's reply ----------
# Try to split after the last occurrence of the user content
reply = decoded
try:
    idx = decoded.rfind(prompt_text.strip())
    reply = decoded[idx + len(prompt_text.strip()):].strip() if idx != -1 else decoded
except Exception:
    pass

# ---------- Post-process bullets (optional) ----------
def enforce_bullets(text, count=3):
    lines = [ln for ln in text.splitlines() if ln.strip().startswith(("-", "•", "*"))]
    lines = lines[:count]
    return "\n".join(lines)

if ENFORCE_BULLETS:
    reply = enforce_bullets(reply, BULLETS)

if STOP_SENTINEL:
    reply = reply.split(STOP_SENTINEL)[0].rstrip()

print("\n=== Assistant ===")
print(reply)
