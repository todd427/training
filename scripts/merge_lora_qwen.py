# scripts/merge_lora_qwen.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, os, json

BASE = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER = "./ckpts/toddric-3b-lora-v0"          # <— your adapter dir
OUT = "./ckpts/toddric-3b-merged-v0"           # <— merged output

os.makedirs(OUT, exist_ok=True)

dtype = (torch.bfloat16 if torch.cuda.is_available()
         and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16)

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=dtype, device_map="auto")
peft = PeftModel.from_pretrained(base, ADAPTER)
merged = peft.merge_and_unload()

merged.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)

gen_cfg = {
  "temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.05,
  "max_new_tokens": 256, "min_new_tokens": 16,
  "pad_token_id": tok.pad_token_id, "eos_token_id": tok.eos_token_id
}
with open(os.path.join(OUT, "generation_config.json"), "w", encoding="utf-8") as f:
    json.dump(gen_cfg, f, ensure_ascii=False, indent=2)

print(f"Merged -> {OUT}")
