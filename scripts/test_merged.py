# scripts/test_merged.py
import os, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.getenv("TODDRIC_MODEL", "/home/todd/training/ckpts/toddric-1_5b-merged-v1")  # <- your path

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
device = torch.device("cuda:0")
dtype = torch.bfloat16  # 40-series friendly; if it errors, use torch.float16

tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=dtype, low_cpu_mem_usage=True
).to(device).eval()
try: model.config.attn_implementation = "sdpa"
except: pass

messages = [
  {"role":"system","content":"You are Toddric — pragmatic, wry, helpful. 1–2 sentences, SMS-length."},
  {"role":"user","content":"Introduce yourself in two short sentences as Toddric."}
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt").to(device)

t0=time.time()
with torch.inference_mode():
  out = model.generate(
      **inputs, max_new_tokens=48, temperature=0.3, top_p=0.9,
      do_sample=True, repetition_penalty=1.05, eos_token_id=tok.eos_token_id
  )
torch.cuda.synchronize()
print(tok.decode(out[0], skip_special_tokens=True))
print("OK in %.2fs" % (time.time()-t0))

