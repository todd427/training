#!/usr/bin/env python3
import csv, json, argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel

DEFAULT_PROMPTS = [
    "Rewrite this email to be friendlier but still professional.",
    "Give me three bullets on why sequence packing helps SFT.",
    "Summarize pros/cons of LoRA vs full fine-tune in 6 lines.",
    "Introduce yourself as Toddric in one stylish paragraph.",
    "Draft a polite follow-up asking for an update on a deliverable."
]

def load_base(model_id, bf16=True):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16 if bf16 else torch.float16
    )
    return tok, model

def load_ft(base_model_id, ft_model_or_adapter, bf16=True, is_adapter=False):
    tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if is_adapter:
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id, device_map="auto",
            dtype=torch.bfloat16 if bf16 else torch.float16
        )
        model = PeftModel.from_pretrained(base, ft_model_or_adapter).to(base.device)
    else:
        # merged full model
        model = AutoModelForCausalLM.from_pretrained(
            ft_model_or_adapter, device_map="auto",
            dtype=torch.bfloat16 if bf16 else torch.float16
        )
    return tok, model

def chat_generate(tok, model, messages, do_sample=True, temperature=0.7, top_p=0.9,
                  max_new_tokens=220, repetition_penalty=1.05):
    # Use each tokenizer’s native chat template
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    gen_cfg = GenerationConfig(
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    with torch.no_grad():
        out = model.generate(**inputs, generation_config=gen_cfg)
    text = tok.decode(out[0], skip_special_tokens=True).strip()
    return text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    # One of these for the fine-tuned model:
    ap.add_argument("--ft_merged", type=str, help="Path or repo id of merged full model.")
    ap.add_argument("--ft_adapter", type=str, help="Path to PEFT adapter dir (use with --base_model).")
    ap.add_argument("--prompts_file", type=Path)
    ap.add_argument("--out", type=Path, default=Path("eval_results_llama"))
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--max_new", type=int, default=220)
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    prompts = DEFAULT_PROMPTS
    if args.prompts_file:
        prompts = [p.strip() for p in args.prompts_file.read_text(encoding="utf-8").splitlines() if p.strip()]

    # Load base
    base_tok, base_model = load_base(args.base_model, bf16=args.bf16)

    # Load fine-tuned (merged or adapter)
    if args.ft_merged:
        ft_tok, ft_model = load_ft(args.base_model, args.ft_merged, bf16=args.bf16, is_adapter=False)
        ft_label = f"Fine-tuned (merged) [{args.ft_merged}]"
    elif args.ft_adapter:
        ft_tok, ft_model = load_ft(args.base_model, args.ft_adapter, bf16=args.bf16, is_adapter=True)
        ft_label = f"Fine-tuned (adapter) [{args.ft_adapter}]"
    else:
        raise SystemExit("Provide either --ft_merged or --ft_adapter")

    do_sample = not args.greedy
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []

    for p in prompts:
        print("="*100)
        print("Prompt:\n" + p)
        print("="*100)

        sys_msg = {"role": "system", "content": "You are a concise, helpful assistant with Toddric tone when appropriate. Avoid signatures or contact info."}
        user_msg = {"role": "user", "content": p}

        base = chat_generate(base_tok, base_model, [sys_msg, user_msg],
                             do_sample=do_sample, max_new_tokens=args.max_new)
        ft = chat_generate(ft_tok, ft_model, [sys_msg, user_msg],
                           do_sample=do_sample, max_new_tokens=args.max_new)

        print("\n[Baseline]\n")
        print(base)
        print(f"\n[{ft_label}]\n")
        print(ft)

        rows.append({"prompt": p, "baseline": base, "fine_tuned": ft})

    # Save CSV & JSON
    csv_path = args.out / "eval_results_llama.csv"
    json_path = args.out / "eval_results_llama.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["prompt", "baseline", "fine_tuned"])
        w.writeheader(); w.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[✓] Saved {csv_path} and {json_path}")

if __name__ == "__main__":
    main()

