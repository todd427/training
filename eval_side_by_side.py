#!/usr/bin/env python3
import argparse, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-jsonl", required=True, help="JSONL from eval_generate (base)")
    ap.add_argument("--toddric-jsonl", required=True, help="JSONL from eval_generate (adapter/merged)")
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    base = [json.loads(l) for l in Path(args.base_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    todd = [json.loads(l) for l in Path(args.toddric_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(base)==len(todd), "Mismatched rows; run on same prompts."

    lines = ["| # | Prompt | Base Output | Toddric Output |",
             "|---:|---|---|---|"]
    for i, (b, t) in enumerate(zip(base, todd), 1):
        lines.append(f"| {i} | {b['prompt']} | {b['completion'].replace('|','\\|')} | {t['completion'].replace('|','\\|')} |")

    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out_md}")

if __name__ == "__main__":
    main()

