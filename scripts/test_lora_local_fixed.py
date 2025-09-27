# eval/test_lora_local_fixed.py
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE   = os.environ.get("TODDRIC_BASE",   "Qwen/Qwen2.5-1.5B-Instruct")
ADAPT  = os.environ.get("TODDRIC_ADAPT",  "/home/todd/training/ckpts/toddric-1_5b-lora-v1")  # <- your LoRA dir
MAX_NEW = int(os.environ.get("MAX_NEW", "160"))

print("torch.cuda.is_available():", torch.cuda.is_available())

# Choose dtype/device deliberately; 1050 Ti can't do bfloat16
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device_map = "auto" if torch.cuda.is_available() else None  # CPU otherwise

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=dtype,
    device_map=device_map
)

model = PeftModel.from_pretrained(base, ADAPT)
model.eval()

# quick confirmation: do we have LoRA modules?
has_lora = any("lora" in n.lower() for n, _ in model.named_modules())
print("LoRA modules attached?:", has_lora)

# Use the model's chat template (critical for Qwen-style instruct models)
messages = [
    {"role": "system", "content": "You are Toddric: friendly, concise, SMS-capable. Stay on topic."},
    {"role": "user", "content": "In one short paragraph, introduce yourself politely as Toddric."}
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tok(prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW,
        temperature=0.6,
        top_p=0.9,
        do_sample=True
    )

print(tok.decode(out[0], skip_special_tokens=True))

