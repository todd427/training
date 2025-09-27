# merge_lora_1b.py
import torch, os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "Qwen/Qwen2.5-1.5B-Instruct"          # adjust if needed
ADAPTER = "/home/todd/training/ckpts/toddric-1_5b-lora-v1"
OUTDIR = "/home/todd/training/ckpts/toddric-1_5b-merged-v1"     # new folder

os.makedirs(OUTDIR, exist_ok=True)

tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.float16, device_map="auto"
)
peft = PeftModel.from_pretrained(base, ADAPTER)
merged = peft.merge_and_unload()              # ← bake LoRA into weights

tok.save_pretrained(OUTDIR)
merged.save_pretrained(OUTDIR)
print("Saved merged model to", OUTDIR)
