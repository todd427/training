#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, time, argparse, datetime, subprocess, sys, tempfile, textwrap

DEFAULT_PROMPTS = [
    "Give me a 2-tweet announcement for Toddric’s alpha in my voice.",
    "Briefly: how to mount a CIFS share on Ubuntu; include the exact command.",
    "Summarize the L.A. Witch series tone in 1 paragraph.",
]

TEMPLATE = r"""
import os, json, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

mid = {MID!r}
attn = {ATTN!r}
prompts = {PROMPTS!r}
max_new = {MAX_NEW}

tok=AutoTokenizer.from_pretrained(mid, use_fast=True)
tok.padding_side="left"; tok.pad_token = tok.pad_token or tok.eos_token
m=AutoModelForCausalLM.from_pretrained(
    mid, device_map={{"":0}},
    torch_dtype=(torch.bfloat16 if torch.cuda.get_device_capability(0)[0]>=8 else torch.float16),
    attn_implementation=attn, low_cpu_mem_usage=True
)
m.eval(); m.config.use_cache=True; m.generation_config.pad_token_id=tok.pad_token_id

rows=[]
@torch.inference_mode()
def run_one(p):
    t=tok.apply_chat_template([{{"role":"system","content":"You are a helpful assistant."}},
                               {{"role":"user","content":p}}], tokenize=False, add_generation_prompt=True)
    ids=tok([t], return_tensors="pt").to(m.device)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0=time.time()
    out=m.generate(**ids, max_new_tokens=max_new, do_sample=False, use_cache=True, eos_token_id=tok.eos_token_id)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    dt=time.time()-t0
    new=int(out.shape[-1]-ids["input_ids"].shape[-1])
    reply=tok.decode(out[0, -new:], skip_special_tokens=True).strip()
    tps=round(new/dt,2) if dt>0 else 0.0
    return dict(latency_s=round(dt,2), new_tokens=new, tokens_per_s=tps, text=reply)

for i,p in enumerate(prompts,1):
    r=run_one(p); r["i"]=i; r["prompt"]=p
    rows.append(r)
print(json.dumps(rows, ensure_ascii=False))
"""

def run_one_model(mid, attn, prompts, max_new):
    code = TEMPLATE.format(MID=mid, ATTN=attn, PROMPTS=prompts, MAX_NEW=max_new)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return json.loads(r.stdout.strip())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--other", required=True)
    ap.add_argument("--base-attn",  default="sdpa",  choices=["sdpa","eager"])
    ap.add_argument("--other-attn", default="eager", choices=["sdpa","eager"])
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--prompts", default="")
    ap.add_argument("--outdir",  default="")
    args=ap.parse_args()

    # Load prompts
    if args.prompts and os.path.exists(args.prompts):
        with open(args.prompts, encoding="utf-8") as f:
            prompts=[ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    else:
        prompts=DEFAULT_PROMPTS

    outdir = args.outdir or f"eval/run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(outdir, exist_ok=True)
    out_jsonl = os.path.join(outdir, "sxs.jsonl")

    print("[base] spawning…"); base_rows = run_one_model(args.base, args.base_attn, prompts, args.max_new)
    print("[other] spawning…"); other_rows= run_one_model(args.other, args.other_attn, prompts, args.max_new)

    rows=[]
    for b,o in zip(base_rows, other_rows):
        rows.append({
            "i": b["i"], "prompt": b["prompt"],
            "base_model": args.base,  "base_attn": args.base_attn,
            "base_latency_s": b["latency_s"], "base_new_tokens": b["new_tokens"], "base_tokens_per_s": b["tokens_per_s"], "base_text": b["text"],
            "other_model": args.other, "other_attn": args.other_attn,
            "other_latency_s": o["latency_s"], "other_new_tokens": o["new_tokens"], "other_tokens_per_s": o["tokens_per_s"], "other_text": o["text"],
        })
        print(f"[{b['i']}/{len(prompts)}] base {b['latency_s']}s ({b['new_tokens']} new, {b['tokens_per_s']} t/s) | "
              f"other {o['latency_s']}s ({o['new_tokens']} new, {o['tokens_per_s']} t/s)")

    with open(out_jsonl, "w", encoding="utf-8") as w:
        for r in rows: w.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("[done]", out_jsonl)

if __name__ == "__main__":
    main()
