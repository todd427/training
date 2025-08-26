# eval/compare_by_index.py
import json, sys

base_path   = "eval/run_20250826/smoke_base_qwen.jsonl"
adapt_path  = "eval/run_20250826/smoke_adapter_qwen_hardened.jsonl"

with open(base_path,"r",encoding="utf-8") as fb, open(adapt_path,"r",encoding="utf-8") as fa:
    base_lines  = [json.loads(x) for x in fb]
    adapt_lines = [json.loads(x) for x in fa]

n = max(len(base_lines), len(adapt_lines))
for i in range(n):
    print("="*80)
    print(f"PAIR #{i+1}")
    # prompt (best-effort from first user message)
    def up(obj):
        for m in obj.get("messages",[]):
            if m.get("role")=="user": return m.get("content")
        return None

    b = base_lines[i]  if i < len(base_lines)  else {}
    a = adapt_lines[i] if i < len(adapt_lines) else {}

    print("PROMPT:", up(b) or up(a) or "(unknown)")

    bo = (b.get("output") or "").strip()
    ao = (a.get("output") or "").strip()

    print("\n--- BASE ---")
    print(bo if bo else "(no output)")
    print(f"[latency {b.get('latency_s','?')}]")

    print("\n--- ADAPTER (toddric) ---")
    print(ao if ao else "(no output)")
    print(f"[latency {a.get('latency_s','?')}]")
