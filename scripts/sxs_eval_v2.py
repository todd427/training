#!/usr/bin/env python3
# scripts/sxs_eval_v3.py
import os, json, time, argparse, datetime, gc, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_PROMPTS = [
    "Give me a 2-tweet announcement for Toddric’s alpha in my voice.",
    "Briefly: how to mount a CIFS share on Ubuntu; include the exact command.",
    "Summarize the L.A. Witch series tone in 1 paragraph.",
]

def pick_dtype():
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32

def load_model(mid: str, attn_impl: str):
    tok = AutoTokenizer.from_pretrained(mid, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        mid,
        device_map={"": 0},
        torch_dtype=pick_dtype(),
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.config.use_cache = True
    if getattr(model.config, "sliding_window", None):
        model.config.sliding_window = None
    gcfg = model.generation_config
    gcfg.pad_token_id = tok.pad_token_id
    gcfg.do_sample = False; gcfg.temperature = None; gcfg.top_p = None; gcfg.top_k = None; gcfg.num_beams = 1
    print(f"== {mid} ==")
    print("hf_device_map:", getattr(model, "hf_device_map", None))
    print("use_cache:", model.config.use_cache, "| dtype:", next(model.parameters()).dtype, "| attn:", attn_impl)
    return tok, model

@torch.inference_mode()
def run_one(tok, model, user_text: str, max_new: int):
    prompt = tok.apply_chat_template(
        [{"role":"system","content":"You are a helpful assistant."},
         {"role":"user","content":user_text}],
        tokenize=False, add_generation_prompt=True)
    ids = tok([prompt], return_tensors="pt").to(model.device)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.time()
    out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                         use_cache=True, eos_token_id=tok.eos_token_id)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    dt = time.time() - t0
    new = int(out.shape[-1] - ids["input_ids"].shape[-1])
    reply = tok.decode(out[0, -new:], skip_special_tokens=True).strip()
    tps = round(new / dt, 2) if dt > 0 else 0.0
    return reply, round(dt,2), new, tps

def read_prompts(path):
    if not path or not os.path.exists(path): return None
    lines=[]
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln=ln.strip()
            if ln and not ln.startswith("#"): lines.append(ln)
    return lines or None

def free(model):
    try:
        del model
    except: pass
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--other", required=True)
    ap.add_argument("--prompts", default="")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--outdir", default="")
    ap.add_argument("--base-attn",  default="sdpa",  choices=["sdpa","eager"])
    ap.add_argument("--other-attn", default="eager", choices=["sdpa","eager"])
    ap.add_argument("--sequential", action="store_true", help="Load models one at a time (recommended on 8–12 GB GPUs)")
    args = ap.parse_args()

    prompts = read_prompts(args.prompts) or DEFAULT_PROMPTS
    outdir = args.outdir or f"eval/run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(outdir, exist_ok=True)
    out_jsonl = os.path.join(outdir, "sxs.jsonl")

    rows=[]
    if args.sequential or (torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory <= 13_000_000_000):
        # sequential: base per prompt, free; then other per prompt, free
        print("[mode] sequential loading")
        # BASE pass
        tb, mb = load_model(args.base, args.base_attn)
        for i,p in enumerate(prompts,1):
            b_txt,b_t,b_new,b_tps = run_one(tb, mb, p, args.max_new)
            rows.append({"i":i,"prompt":p,"base_model":args.base,"base_attn":args.base_attn,
                         "base_latency_s":b_t,"base_new_tokens":b_new,"base_tokens_per_s":b_tps,"base_text":b_txt})
            print(f"[base {i}/{len(prompts)}] {b_t}s ({b_new} new, {b_tps} t/s)")
        free(mb)
        # OTHER pass
        to, mo = load_model(args.other, args.other_attn)
        for i,p in enumerate(prompts,1):
            o_txt,o_t,o_new,o_tps = run_one(to, mo, p, args.max_new)
            # merge into same row index
            rows[i-1].update({"other_model":args.other,"other_attn":args.other_attn,
                              "other_latency_s":o_t,"other_new_tokens":o_new,"other_tokens_per_s":o_tps,"other_text":o_txt})
            print(f"[other {i}/{len(prompts)}] {o_t}s ({o_new} new, {o_tps} t/s)")
        free(mo)
    else:
        # parallel load (only if you have plenty of VRAM)
        print("[mode] parallel loading (VRAM >= 16 GB recommended)")
        tb, mb = load_model(args.base, args.base_attn)
        to, mo = load_model(args.other, args.other_attn)
        for i,p in enumerate(prompts,1):
            b_txt,b_t,b_new,b_tps = run_one(tb, mb, p, args.max_new)
            o_txt,o_t,o_new,o_tps = run_one(to, mo, p, args.max_new)
            rows.append({"i":i,"prompt":p,
                         "base_model":args.base,"base_attn":args.base_attn,"base_latency_s":b_t,"base_new_tokens":b_new,"base_tokens_per_s":b_tps,"base_text":b_txt,
                         "other_model":args.other,"other_attn":args.other_attn,"other_latency_s":o_t,"other_new_tokens":o_new,"other_tokens_per_s":o_tps,"other_text":o_txt})
        free(mb); free(mo)

    with open(out_jsonl,"w",encoding="utf-8") as w:
        for r in rows: w.write(json.dumps(r, ensure_ascii=False)+"\n")
    print("[done]", out_jsonl)

if __name__ == "__main__":
    main()
