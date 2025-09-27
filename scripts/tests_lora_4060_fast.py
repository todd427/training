# scripts/test_lora_4060_fast.py
import os, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE   = os.environ.get("TODDRIC_BASE", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPT  = os.environ.get("TODDRIC_ADAPT", "/home/todd/training/ckpts/toddric-1_5b-lora-v1")  # <- set this
MAX_NEW = int(os.environ.get("MAX_NEW", "48"))  # SMS-sized for speed

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

assert torch.cuda.is_available(), "CUDA not available — 4060 not visible to PyTorch?"
device = torch.device("cuda:0")

# bf16 preferred on 40-series; fallback to fp16 if env/driver blocks it
dtype = torch.bfloat16
try:
    x = torch.empty(1, dtype=dtype, device=device)  # quick capability check
except Exception:
    dtype = torch.float16

print("CUDA device:", torch.cuda.get_device_name(0))
print("Using dtype:", dtype)
print("Before load, CUDA mem (MB):", torch.cuda.memory_allocated() / (1024*1024))

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)

# Force full model on GPU (no device_map='auto' offload)
base = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=dtype,
    low_cpu_mem_usage=True
).to(device).eval()

# Prefer SDPA; Transformers uses it automatically on recent PyTorch,
# but we can hint via config:
try:
    base.config.attn_implementation = "sdpa"  # safe even if not used
except Exception:
    pass

# Attach LoRA
model = PeftModel.from_pretrained(base, ADAPT).to(device).eval()

# Quick mem print
torch.cuda.synchronize()
print("After load, CUDA mem (MB):", torch.cuda.memory_allocated() / (1024*1024))

# Use the chat template (critical for Qwen Instruct)
messages = [
    {"role": "system", "content": "You are Toddric: friendly, concise, SMS-capable. Stay on topic."},
    {"role": "user", "content": "In two short sentences, introduce yourself politely as Toddric."}
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tok(prompt, return_tensors="pt").to(device)

t0 = time.time()
with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW,
        temperature=0.4,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.05
    )
torch.cuda.synchronize()
dt = time.time() - t0

print(tok.decode(out[0], skip_special_tokens=True))
print(f"GEN {MAX_NEW} tokens in {dt:.2f}s")

