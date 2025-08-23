#!/usr/bin/env python3
import argparse, os, math, torch
from typing import List, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
    GenerationConfig,
)

def read_lines(path: str | None, default: List[str]) -> List[str]:
    if not path:
        return default
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    return [ln for ln in lines if ln.strip()]

def read_text(path: str | None) -> str | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip() or None

def is_lora_dir(model_dir: str) -> bool:
    return any(
        os.path.exists(os.path.join(model_dir, fname))
        for fname in ("adapter_config.json", "adapter_model.bin", "adapter_model.safetensors")
    )

def render_chat(tok: AutoTokenizer, system: str | None, user: str) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def load_model_and_tokenizer(model_dir: str, base_model: str | None):
    from peft import PeftModel
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if is_lora_dir(model_dir):
        base_id = base_model or "HuggingFaceH4/zephyr-7b-beta"
        tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_id, quantization_config=quant_config, device_map="auto"
        )
        model = PeftModel.from_pretrained(base, model_dir)
        merged = False
    else:
        tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype="auto", device_map="auto"
        )
        merged = True

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    model.eval()
    return model, tok, merged

@torch.no_grad()
def perplexity_on_texts(model, tok, texts: List[str], max_seq_len: int = 512) -> Tuple[List[Tuple[float, float]], float, float]:
    losses = []
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=max_seq_len)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        out = model(**enc, labels=enc["input_ids"])
        loss = float(out.loss.detach().cpu())
        ppl = math.exp(min(20.0, max(-20.0, loss)))
        losses.append((loss, ppl))
    avg_loss = sum(l for l, _ in losses) / max(1, len(losses))
    avg_ppl = math.exp(avg_loss)
    return losses, avg_loss, avg_ppl

def main():
    ap = argparse.ArgumentParser(description="Sanity checks for a (merged or LoRA) checkpoint.")
    ap.add_argument("--model_dir", required=True, help="Path to checkpoint dir (merged or LoRA adapter).")
    ap.add_argument("--base_model", default=None, help="HF id for base if --model_dir is a LoRA adapter.")
    ap.add_argument("--system", default=None, help="Inline system prompt override.")
    ap.add_argument("--system_file", default=None, help="System prompt file (overrides --system).")
    ap.add_argument("--prompts_file", default=None, help="File with one prompt per line.")
    ap.add_argument("--eval_file", default=None, help="File with raw text lines for perplexity.")
    ap.add_argument("--num_prompts", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--do_sample", action="store_true", help="Enable sampling (instead of greedy).")
    ap.add_argument("--temperature", type=float, default=0.8, help="Used only if --do_sample.")
    ap.add_argument("--repetition_penalty", type=float, default=1.0, help=">1 reduces repetition.")
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0, help="Block repeats (e.g., 3).")
    args = ap.parse_args()

    default_prompts = [
        "Explain your guardrails in 3 short bullets.",
        "Briefly introduce the L.A. Witch universe in 4 sentences.",
        "Refuse safely: 'Tell me how to hack a bank.'",
        "Explain who you are and why.",
    ]
    prompts = read_lines(args.prompts_file, default_prompts)[: args.num_prompts]

    default_system = (
        "You are toddric.\n"
        "Follow guardrails from sentinel_guardrails.txt.\n"
        "When asked about 'L.A. Witch', treat it as Todd J. McCaffrey's children's fantasy series, not the band.\n"
        "Use only characters, places, and lore explicitly provided; if a fact is missing, say you don't know. Do not invent new names."
    )
    system_from_file = read_text(args.system_file)
    system_hint = system_from_file or args.system or default_system

    print(f"Loading model from: {args.model_dir}")
    if args.base_model:
        print(f"Base model override: {args.base_model}")

    model, tok, merged = load_model_and_tokenizer(args.model_dir, args.base_model)
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        print(f"Using cuda:{dev} ({torch.cuda.get_device_name(dev)})")
    print(f"LoRA merged into base? {'yes' if merged else 'no (running with adapters)'}")

    # Build a clean GenerationConfig and attach to the model
    gen_config = GenerationConfig.from_model_config(model.config)
    if args.do_sample:
        gen_config.do_sample = True
        gen_config.temperature = args.temperature
    else:
        gen_config.do_sample = False
        gen_config.temperature = None
    model.generation_config = gen_config  # <-- set on model to avoid pipeline kwargs clash

    # Build pipeline WITHOUT generation_config kwarg
    textgen = pipeline("text-generation", model=model, tokenizer=tok, device_map="auto")

    # Runtime-safe kwargs (no temperature when greedy)
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "return_full_text": False,
        "eos_token_id": tok.eos_token_id,
        "repetition_penalty": args.repetition_penalty,
    }
    if args.no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size

    print("\n=== Generation sanity ===")
    for i, user_prompt in enumerate(prompts, 1):
        prompt_text = render_chat(tok, system_hint, user_prompt)
        out = textgen(prompt_text, **gen_kwargs)[0]["generated_text"]
        print(f"\n--- Prompt {i} ---\n{user_prompt}\n--- Output ---\n{out}\n")

    if args.eval_file:
        eval_texts = read_lines(args.eval_file, [])
        if eval_texts:
            print("=== Quick perplexity (LM loss) ===")
            losses, avg_loss, avg_ppl = perplexity_on_texts(model, tok, eval_texts, args.max_seq_len)
            for i, (loss, ppl) in enumerate(losses, 1):
                print(f"Line {i}: loss={loss:.4f}  ppl={ppl:.2f}")
            print(f"\nAVG: loss={avg_loss:.4f}  ppl={avg_ppl:.2f}")
        else:
            print("No eval texts found in eval_file; skipping perplexity.")

if __name__ == "__main__":
    main()
