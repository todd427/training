#!/usr/bin/env python
import argparse, torch, sys
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
from peft import PeftModel

def load(base, adapter=None, fourbit=True):
    bnb = None
    if fourbit:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(base, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base, device_map="auto",
                                                 trust_remote_code=True, quantization_config=bnb)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    return tok, model

def build_prompt(tok, messages):
    # Uses the model’s own chat template
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="e.g., Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default="", help="path to LoRA folder")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--no_4bit", action="store_true")
    args = ap.parse_args()

    tok, model = load(args.base, args.adapter or None, fourbit=not args.no_4bit)
    sys_msg = {"role": "system", "content": "You are Toddric, concise, helpful, and technically capable."}
    history = [sys_msg]

    print("Chat ready. Type 'exit' or Ctrl-C to quit.\n")
    while True:
        try:
            user = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nbye.")
            break
        if user.lower() in {"exit","quit","/q"}:
            break
        history.append({"role": "user", "content": user})

        prompt = build_prompt(tok, history)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p
            )
        # decode last turn (streamer already printed, but keep a clean assistant message in history)
        reply = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print()  # newline after stream
        history.append({"role": "assistant", "content": reply.strip()})

if __name__ == "__main__":
    main()
