# scripts/merge_lora.py
#!/usr/bin/env python3
import os, argparse, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--out",  required=True)
    a=ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.base, use_fast=True)

    base = AutoModelForCausalLM.from_pretrained(
        a.base,
        device_map={"":0},
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0]>=8 else torch.float16),
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    base.config.use_cache = True
    if getattr(base.config, "sliding_window", None):
        base.config.sliding_window = None
    base.gradient_checkpointing_disable()

    peft = PeftModel.from_pretrained(base, a.lora, device_map={"":0})
    merged = peft.merge_and_unload()  # folds LoRA into base weights

    os.makedirs(a.out, exist_ok=True)
    merged.save_pretrained(a.out, safe_serialization=True)
    tok.save_pretrained(a.out)

    # patch config just in case
    cfg = os.path.join(a.out, "config.json")
    j = json.load(open(cfg))
    j["use_cache"] = True
    if "sliding_window" in j: j["sliding_window"] = None
    json.dump(j, open(cfg,"w"), indent=2)
    print("[done] merged →", a.out)

if __name__ == "__main__":
    main()
