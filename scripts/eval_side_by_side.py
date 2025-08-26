import argparse
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=str, default="eval/prompts.jsonl",
                        help="JSONL file with eval prompts (default: eval/prompts.jsonl)")
    parser.add_argument("--out", type=str, default="eval/toddric_eval.md",
                        help="Output markdown file (default: eval/toddric_eval.md)")
    parser.add_argument("--base", type=str, required=True,
                        help="Base model ID (e.g. openai/oss-gpt-20b)")
    parser.add_argument("--adapter", type=str, required=True,
                        help="Path to SFT checkpoint or adapter")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--dtype", type=str, choices=["bf16","fp16","fp32"], default="bf16")
    args = parser.parse_args()

    # Load models
    print("[*] Loading base model...")
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype={"bf16":"bfloat16","fp16":"float16","fp32":"float32"}[args.dtype],
        device_map="auto"
    )
    base_pipe = pipeline("text-generation", model=base_model, tokenizer=tok)

    print("[*] Loading Toddric adapter...")
    toddric_model = AutoModelForCausalLM.from_pretrained(
        args.adapter,
        torch_dtype={"bf16":"bfloat16","fp16":"float16","fp32":"float32"}[args.dtype],
        device_map="auto"
    )
    toddric_pipe = pipeline("text-generation", model=toddric_model, tokenizer=tok)

    # Read prompts
    with open(args.prompts, "r", encoding="utf-8") as f:
        prompts = [json.loads(line)["prompt"] for line in f if line.strip()]

    # Generate
    rows = []
    for p in prompts:
        base_out = base_pipe(p, max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature, top_p=args.top_p)[0]["generated_text"]
        toddric_out = toddric_pipe(p, max_new_tokens=args.max_new_tokens,
                                   temperature=args.temperature, top_p=args.top_p)[0]["generated_text"]

        rows.append((p, base_out.strip(), toddric_out.strip()))

    # Write results in Markdown
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out_f:
        out_f.write("| Prompt | Base | Toddric-SFT |\n")
        out_f.write("|--------|------|-------------|\n")
        for p, base_out, toddric_out in rows:
            out_f.write(f"| {p} | {base_out.replace('|','/')} | {toddric_out.replace('|','/')} |\n")

    print(f"[*] Wrote results to {args.out}")

if __name__ == "__main__":
    main()
