# test_lora_local.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "Qwen/Qwen2.5-1.5B-Instruct"          # <- your base (adjust if different)
ADAPTER = "/home/todd/training/ckpts/toddric-1_5b-lora-v1"     # <- your LoRA folder

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

prompt = "You are Jimi Hendrix. In one paragraph, introduce yourself politely.\n"
inputs = tok(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=160,
        temperature=0.6,
        top_p=0.9
    )
print(tok.decode(out[0], skip_special_tokens=True))

