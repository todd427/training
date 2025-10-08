#!/bin/bash
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, pathlib

base = "meta-llama/Llama-3.1-8B-Instruct"
adapter = "ckpts/toddric-llama-8B-lora-v1clean"
out_dir = "ckpts/toddric-llama-8B-merged-v1"

print(f"[merge] base={base} adapter={adapter}")
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()
model.save_pretrained(out_dir, safe_serialization=True)
tok.save_pretrained(out_dir)

print(f"[done] merged model saved to {out_dir}")
PY

