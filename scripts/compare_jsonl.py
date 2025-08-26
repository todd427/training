# eval/compare_jsonl.py
import json

def load(path):
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            r = json.loads(line)
            # try to find first user message
            prompt = None
            for m in r.get("messages", []):
                if m.get("role") == "user":
                    prompt = m.get("content")
                    break
            # fallback: use index if no prompt found
            if not prompt:
                prompt = f"prompt_{i}"
            rows[prompt] = r
    return rows

base = load("eval/run_20250826/smoke_base_qwen.jsonl")
adapt = load("eval/run_20250826/smoke_adapter_qwen.jsonl")

for prompt in base:
    print("="*80)
    print("PROMPT:", prompt)
    print("--- BASE ---")
    print(base[prompt]["output"].strip())
    print(f"[latency {base[prompt].get('latency_s','?')}s]")
    print("--- ADAPTER ---")
    if prompt in adapt:
        print(adapt[prompt]["output"].strip())
        print(f"[latency {adapt[prompt].get('latency_s','?')}s]")
    else:
        print("(no adapter output)")
