#!/usr/bin/env python3
import argparse, json, os, time, random
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_model(base_or_merged: str, adapter: str|None, dtype="bfloat16"):
    torch_dtype = {"bf16":"bfloat16","fp16":"float16","fp32":"float32"}[dtype]
    tok = AutoTokenizer.from_pretrained(base_or_merged, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(base_or_merged, torch_dtype=getattr(__import__("torch"), torch_dtype), device_map="auto")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    return tok, model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="Text file: one prompt per line")
    ap.add_argument("--out", required=True, help="Output JSONL")
    ap.add_argument("--base", required=True, help="Base or merged model id/path")
    ap.add_argument("--adapter", default=None, help="LoRA adapter repo/path (omit if merged)")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--dtype", choices=["bf16","fp16","fp32"], default="bf16")
    args = ap.parse_args()

    prompts = [ln.strip() for ln in Path(args.prompts).read_text(encoding="utf-8").splitlines() if ln.strip()]
    tok, model = load_model(args.base, args.adapter, dtype=args.dtype)

    os.makedirs(Path(args.out).parent, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i, p in enumerate(prompts, 1):
            inputs = tok(p, return_tensors="pt").to(model.device)
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            text = tok.decode(out[0], skip_special_tokens=True)
            rec = {"prompt": p, "completion": text[len(p):].strip(), "ts": int(time.time())}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(prompts)}] done")

if __name__ == "__main__":
    main()

