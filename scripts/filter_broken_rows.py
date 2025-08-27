#!/usr/bin/env python3
import json, argparse, os, re

def ok(o):
    if not isinstance(o, dict): return False
    m = o.get("messages")
    if not isinstance(m, list) or len(m) < 3: return False
    roles = [x.get("role") for x in m if isinstance(x, dict)]
    if "user" not in roles or "assistant" not in roles: return False
    # basic content check
    for x in m:
        if not isinstance(x.get("content",""), str) or not x.get("content","").strip():
            return False
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()

    keep=drop=0
    with open(os.path.expanduser(args.inp), encoding="utf-8") as f, \
         open(os.path.expanduser(args.out), "w", encoding="utf-8") as w:
        for ln in f:
            try:
                o=json.loads(ln)
            except:
                drop+=1; continue
            if ok(o):
                w.write(json.dumps(o, ensure_ascii=False)+"\n"); keep+=1
            else:
                drop+=1
    print(f"[filter] kept={keep} dropped={drop} → {args.out}")

if __name__ == "__main__":
    main()
