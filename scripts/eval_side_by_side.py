#!/usr/bin/env python3
import csv, json, argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
FT_MODEL   = "toddie314/toddric-3b-merged-v0"

DEFAULT_PROMPTS = [
    "Explain why the sky is blue in two sentences.",
    "Write a four-line poem about dragons and libraries.",
    "List three steps to make a perfect cup of tea.",
    "Introduce yourself as Toddric, a witty AI assistant, in two sentences.",
    "Summarize the pros and cons of using AI in creative writing."
]

def chat_generate(model_id, messages, do_sample=True, temperature=0.7, top_p=0.9,
                  max_new_tokens=180, repetition_penalty=1.05):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )

    # Build chat-formatted prompt using each model's template
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
    output_ids = model.generate(**inputs, generation_config=gen_cfg)
    text = tok.decode(output_ids[0], skip_special_tokens=True)

    # Extract just the assistant’s final turn if the template includes roles
    # Heuristic: split on '\nassistant' or similar is overkill; the template usually puts the answer at the end.
    # Trim trailing contact lines often learned from SFT:
    for sig in ["@toddwriter.com", "linkedin.com/", "substack.com", "Twitter", "icloud.com", "http://", "https://"]:
        if sig in text:
            text = text.split(sig)[0].rstrip()
    return text.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--greedy", action="store_true", help="Use greedy decoding.")
    ap.add_argument("--max_new", type=int, default=180)
    ap.add_argument("--out", type=Path, default=Path("eval_results"))
    ap.add_argument("--prompts_file", type=Path, help="Optional text file with one prompt per line.")
    args = ap.parse_args()

    prompts = DEFAULT_PROMPTS
    if args.prompts_file:
        prompts = [p.strip() for p in args.prompts_file.read_text(encoding="utf-8").splitlines() if p.strip()]

    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in prompts:
        print("="*80)
        print("Prompt:\n" + p)
        print("="*80)

        sys_msg = {"role": "system", "content": "You are a concise, helpful assistant. Avoid signatures or contact info."}
        user_msg = {"role": "user", "content": p}

        base = chat_generate(
            BASE_MODEL, [sys_msg, user_msg],
            do_sample=not args.greedy, max_new_tokens=args.max_new
        )
        ft = chat_generate(
            FT_MODEL, [sys_msg, user_msg],
            do_sample=not args.greedy, max_new_tokens=args.max_new
        )

        print("\n[Baseline: Qwen2.5-3B-Instruct]\n")
        print(base)
        print("\n[Fine-tuned: Toddric-3B-Merged]\n")
        print(ft)

        rows.append({"prompt": p, "baseline": base, "fine_tuned": ft})

    # Save CSV & JSON
    csv_path = args.out / "eval_results.csv"
    json_path = args.out / "eval_results.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["prompt", "baseline", "fine_tuned"])
        w.writeheader(); w.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[✓] Saved {csv_path} and {json_path}")

if __name__ == "__main__":
    main()
