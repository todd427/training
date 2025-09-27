#!/usr/bin/env python3
import os, json, argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path or HF id of your merged fp16/bf16 model")
    ap.add_argument("--out", required=True, help="Output dir for the bnb-4bit model")
    ap.add_argument("--attn", default="eager", choices=["eager","sdpa"])
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16","float16"])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 4-bit config (good defaults for Qwen2.5-3B)
    qconf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=(torch.bfloat16 if args.dtype=="bfloat16" else torch.float16),
    )

    tok = AutoTokenizer.from_pretrained(args.src, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print("[load] quantized 4-bit view of", args.src)
    model = AutoModelForCausalLM.from_pretrained(
        args.src,
        device_map={"":0},
        quantization_config=qconf,
        attn_implementation=args.attn,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.config.use_cache = True
    model.generation_config.pad_token_id = tok.pad_token_id

    print("[save] writing tokenizer + model (bnb-4bit) →", args.out)
    tok.save_pretrained(args.out)
    # Persist the quantized weights and config
    model.save_pretrained(args.out, safe_serialization=True)

    # Also drop an explicit quantization_config.json for clarity
    with open(os.path.join(args.out, "quantization_config.json"), "w", encoding="utf-8") as f:
        json.dump(qconf.to_dict(), f, ensure_ascii=False, indent=2)

    print("[done] bnb-4bit export at:", args.out)

if __name__ == "__main__":
    main()
