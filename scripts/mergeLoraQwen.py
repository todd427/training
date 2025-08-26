# scripts/merge_lora_qwen.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, os

BASE = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER = "./ckpts/toddric-qwen2p5-3b-lora-v1"
OUT = "./ckpts/toddric-qwen2p5-3b-merged-v1"

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map="auto")
peft = PeftModel.from_pretrained(base, ADAPTER)
merged = peft.merge_and_unload()

os.makedirs(OUT, exist_ok=True)
merged.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
print("Merged →", OUT)
