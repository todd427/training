# scripts/sxs_eval_v2.py
#!/usr/bin/env python3
import os, json, time, argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPTS = [
  "Give me a 2-tweet announcement for Toddric’s alpha in my voice.",
  "Briefly: how to mount a CIFS share on Ubuntu; include the exact command.",
  "Summarize the L.A. Witch series tone in 1 paragraph."
]

def load(mid):
    tok=AutoTokenizer.from_pretrained(mid, use_fast=True)
    tok.padding_side="left"
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    m=AutoModelForCausalLM.from_pretrained(
        mid,
        device_map={"":0},  # FORCE everything on cuda:0
        torch_dtype=(torch.bfloat16 if torch.cuda.get_device_capability(0)[0]>=8 else torch.float16),
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    # force fast decode flags
    m.config.use_cache=True
    if getattr(m.config, "sliding_window", None):
        m.config.sliding_window=None
    gc = m.generation_config
    gc.pad_token_id = tok.pad_token_id
    print(f"== {mid} ==")
    print("hf_device_map:", getattr(m, "hf_device_map", None))
    print("use_cache:", m.config.use_cache, "| torch_dtype:", next(m.parameters()).dtype)
    return tok,m

def gen(tok,m,prompt,max_new=200):
    msgs=[{"role":"system","content":"You are a helpful assistant."},
          {"role":"user","content":prompt}]
    text=tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids=tok([text], return_tensors="pt").to(m.device)
    torch.cuda.synchronize()
    t0=time.time()
    with torch.inference_mode():
        out=m.generate(**ids, max_new_tokens=max_new, do_sample=False,
                       eos_token_id=tok.eos_token_id, use_cache=True)
    torch.cuda.synchronize()
    new = int(out.shape[-1]-ids["input_ids"].shape[-1])
    return round(time.time()-t0,2), new

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--other", required=True)
    args=ap.parse_args()

    tb,mb=load(args.base)
    to,mo=load(args.other)

    rows=[]
    for p in PROMPTS:
        bt,bn=gen(tb,mb,p); ot,on=gen(to,mo,p)
        rows.append(dict(prompt=p, base_t=bt, other_t=ot, base_new=bn, other_new=on))
        print(rows[-1])
    outdir=f"eval/run_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir,"sxs.jsonl"),"w",encoding="utf-8") as w:
        for r in rows: w.write(json.dumps(r,ensure_ascii=False)+"\n")
    print("[done]", outdir)

if __name__=="__main__":
    main()
