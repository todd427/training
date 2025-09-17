import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel

base_id = "meta-llama/Llama-3.1-8B-Instruct"
adapter_dir = "/home/Projects/toddric/training/ckpts/toddric-llama-8B-lora"

tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

base = AutoModelForCausalLM.from_pretrained(base_id, device_map="auto", dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, adapter_dir)

msgs = [
  {"role": "system", "content": "You are Toddric—precise, helpful, and concise."},
  {"role": "user", "content": "Draft a friendly, concise follow-up asking for the project update by Friday."},
]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt").to(model.device)

gen_cfg = GenerationConfig(do_sample=True, temperature=0.6, top_p=0.9, max_new_tokens=220)
out = model.generate(**inputs, generation_config=gen_cfg)
print(tok.decode(out[0], skip_special_tokens=True))

