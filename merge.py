#!/usr/bin/env python3
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "HuggingFaceH4/zephyr-7b-beta"
adapter = "toddie314/toddric-zephyr7b-lora"
dst = "/workspace/release/zephyr7b-toddric-merged"

tok = AutoTokenizer.from_pretrained(base, use_fast=True)
m = AutoModelForCausalLM.from_pretrained(base, torch_dtype="bfloat16", device_map="auto")
m = PeftModel.from_pretrained(m, adapter)
m = m.merge_and_unload()  # bake LoRA into base
m.save_pretrained(dst, safe_serialization=True)
tok.save_pretrained(dst)
print("Saved merged model to", dst)

