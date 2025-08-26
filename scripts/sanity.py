#!/usr/bin/env python3
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, GenerationConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--greedy", action="store_true",
        help="Use greedy decoding (deterministic). Default is sampling ON."
    )
    args = parser.parse_args()

    model_id = "toddie314/toddric-3b-merged-v0"

    print(f"[*] Loading {model_id} ...")
    tok = AutoTokenizer.from_pretrained(model_id)
    m = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto"
    )

    # Explicit generation config (bypasses baked-in JSON)
    if args.greedy:
        print("[*] Using greedy decoding (deterministic).")
        gen_cfg = GenerationConfig(
            do_sample=False,
            max_new_tokens=200,
            repetition_penalty=1.1,
        )
    else:
        print("[*] Using sampling (temperature=0.7, top_p=0.9).")
        gen_cfg = GenerationConfig(
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=200,
            repetition_penalty=1.1,
        )

    # Example conversation
    msgs = [
        {"role": "system", "content": "You are Toddric, concise, witty, and helpful."},
        {"role": "user", "content": "Give me a two-sentence status update on training."}
    ]

    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(m.device)

    stream = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    m.generate(**inputs, streamer=stream, generation_config=gen_cfg)


if __name__ == "__main__":
    main()
