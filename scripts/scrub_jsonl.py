#!/usr/bin/env python3
"""
scrub_jsonl.py — clean PII & signature lines from training data.

Usage:
  python scripts/scrub_jsonl.py input.jsonl [output.jsonl]

If output is omitted, writes to input_cleaned.jsonl.
"""

import re, json, sys, pathlib

PII_PATTERNS = [
    r"https?://\S+",          # URLs
    r"www\.\S+",              # www.*
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",  # emails
    r"(?:(?:\+?\d{1,3}[-\s]?)?(?:\(?\d{2,4}\)?[-\s]?)?\d{3,4}[-\s]?\d{4})",  # phones
    r"Privacy Policy", r"Terms of Service", r"Unsubscribe",
    r"Regards,", r"Best,", r"Sincerely,", r"Kind regards,", r"--\s*$",  # signatures
]

def line_has_pii(text: str) -> bool:
    for pat in PII_PATTERNS:
        if re.search(pat, text, re.I | re.M):
            return True
    return False

def clean_obj(obj):
    """Recursively drop fields or messages containing PII."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, str):
                if line_has_pii(v):
                    continue
            out[k] = clean_obj(v)
        return out
    elif isinstance(obj, list):
        return [clean_obj(v) for v in obj if not (isinstance(v, str) and line_has_pii(v))]
    else:
        return obj

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    inp = pathlib.Path(sys.argv[1])
    outp = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else inp.with_name(inp.stem + "_cleaned.jsonl")

    kept, dropped = 0, 0
    with inp.open("r", encoding="utf-8") as f, outp.open("w", encoding="utf-8") as g:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            blob = json.dumps(obj, ensure_ascii=False)
            if line_has_pii(blob):
                dropped += 1
                continue
            cleaned = clean_obj(obj)
            g.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[scrub] wrote {kept} clean lines to {outp} (dropped {dropped})")

if __name__ == "__main__":
    main()

