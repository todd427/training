# eval/smoke_qwen_base.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, json, os, time

BASE="Qwen/Qwen2.5-3B-Instruct"
OUTDIR="eval/run_20250826"; os.makedirs(OUTDIR, exist_ok=True)

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map="auto")
prompts = [
  {"role":"user","content":"Give me a 2-tweet announcement for Toddric’s alpha in my voice."},
  {"role":"user","content":"Briefly: how to mount a CIFS share on Ubuntu; include the exact command."},
  {"role":"user","content":"Summarize the L.A. Witch series tone in 1 paragraph."},
]

outs=[]
for m in prompts:
    messages=[{"role":"system","content":"You are Toddric, Todd’s helpful assistant with Todd’s tone."}, m]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    t0=time.time()
    out = model.generate(**ids, max_new_tokens=256, temperature=0.7, top_p=0.9)
    txt = tok.decode(out[0], skip_special_tokens=True)
    outs.append({"messages":messages,"output":txt,"latency_s":round(time.time()-t0,2)})

with open(f"{OUTDIR}/smoke_base_qwen.jsonl","w") as f:
    for o in outs: f.write(json.dumps(o,ensure_ascii=False)+"\n")
print("Saved", f"{OUTDIR}/smoke_base_qwen.jsonl")
