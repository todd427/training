#!/usr/bin/env python3
import argparse, json, re, statistics
from pathlib import Path
def toks(x): return re.findall(r"\w+|\S", x)
def uniq_ratio(seq): 
    return len(set(seq))/max(1,len(seq))
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    args = ap.parse_args()
    rows = [json.loads(l) for l in Path(args.jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    lens = [len(toks(r["completion"])) for r in rows]
    uniqs = [uniq_ratio(toks(r["completion"])) for r in rows]
    print(f"samples={len(rows)}")
    print(f"len_tokens: mean={statistics.mean(lens):.1f} median={statistics.median(lens):.1f}")
    print(f"uniq_ratio: mean={statistics.mean(uniqs):.3f} median={statistics.median(uniqs):.3f}")
if __name__ == "__main__":
    main()

