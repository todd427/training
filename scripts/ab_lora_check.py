# scripts/ab_lora_check.py
import os, time, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE   = os.environ.get("TODDRIC_BASE", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPT  = os.environ.get("TODDRIC_ADAPT", "/home/todd/training/ckpts/toddric-1_5b-lora-v1")  # <<< set this
MAX_NEW = int(os.environ.get("MAX_NEW", "120"))

def show_where(model):
    # Print where the big blocks live
    for name, p in model.named_parameters():
        if any(k in name for k in ("lm_head.weight","model.embed_tokens.weight")):
            print(f"{name}: {str(p.device)} shape={tuple(p.shape)}")
            break

print("torch.cuda.is_available():", torch.cuda.is_available())
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device_map = "auto" if torch.cuda.is_available() else None

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)

def run_messages(model, messages):
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.05
        )
    txt = tok.decode(out[0], skip_special_tokens=True)
    return txt, time.time() - t0

messages = [
    {"role":"system","content":"You are Toddric: concise, helpful, author/engineer vibe, SMS-friendly."},
    {"role":"user","content":"Briefly, what is 'University of Souls' (our project) and what are its tiers?"}
]

# BASE
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=dtype, device_map=device_map)
base.eval()
print("\n[BASE WHERE]")
show_where(base)
b_txt, b_sec = run_messages(base, messages)
print("\n[BASE OUTPUT]\n", b_txt)
print(f"[BASE LATENCY] {b_sec:.2f}s")

# LoRA
lora = PeftModel.from_pretrained(base, ADAPT)
lora.eval()
print("\n[LORA WHERE]")
show_where(lora)
l_txt, l_sec = run_messages(lora, messages)
print("\n[LORA OUTPUT]\n", l_txt)
print(f"[LORA LATENCY] {l_sec:.2f}s")

# Quick diff cue
def sig(s): return set(w.lower().strip(".,:;!?") for w in s.split())
added = sig(l_txt) - sig(b_txt)
print("\n[DIFF] words present in LoRA but not Base (sample):", list(sorted(list(added)))[:30])

