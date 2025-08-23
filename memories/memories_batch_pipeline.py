#!/usr/bin/env python3
"""
Batch memories pipeline (auto-chunks):
TXT dir --> sentences JSONL (memory schema) --> event chunks JSONL --> SFT JSONL (instructions)

For each .txt in INPUT_DIR:
  - Write sentence-level JSONL to OUT_ROOT/sentences/<name>.jsonl
  - Write chunk-level JSONL     to OUT_ROOT/chunks/<name>.jsonl
  - Write SFT instruction JSONL to OUT_ROOT/sft/<name>.jsonl

Usage:
  python memories_batch_pipeline.py INPUT_DIR OUT_ROOT --pattern "*.txt" --min-sent 2 --max-sent 6

Notes:
- Zero dependencies (standard library only).
- Heuristics are lightweight; tune keyword sets as your corpus grows.
"""

import os, re, json, uuid, argparse, fnmatch, sys
from typing import List, Dict, Any

# -------- Heuristics (tuned to age6 + age9; expand as needed) --------
LOCATION_HINTS = {
    # age6 travel & places
    "wilmington", "delaware", "dusseldorf", "germany", "rotterdam",
    "england", "verdun", "heathrow", "united states", "america",
    "vw", "volkswagen",
    # age9 new places
    "manhattan", "long island", "sea cliff", "new york", "ireland", "france",
}
PEOPLE_WHITELIST = {
    # family + recurring
    "Mom", "Mum", "Dad", "Alec", "Gigi",
    # age6
    "Gisela Quante", "Ken", "Ken Estes", "state troopers", "principal", "gardener",
    # age9 (co‑renting / neighbors)
    "Mr. Isbell", "Mrs. Isbell", "Linda Isbell", "Kenny Isbell",
    "Lisa Guy", "Tommy Guy", "Guys",
}
THEME_KEYWORDS = {
    # mobility / travel
    "travel": {"flying","airport","boeing 707","across the atlantic","ferry",
               "europe","england","germany","rotterdam","verdun","vw","volkswagen",
               "station wagon","camped","departure tax"},
    # school & discipline
    "school": {"elementary school","principal","teacher","first grade","second grade",
               "cursive","retrained","held back","repeat first grade"},
    "discipline": {"beating","corporal punishment","principal's office"},
    # injuries / hospitals
    "injury": {"stitches","no anesthetic","hammer","machete","hit","eye","shin","hospital"},
    # family & food
    "family": {"mom","mum","dad","brother","sister","parents","family"},
    "food": {"bread","ice cream","rose extract","mustard soup","cook","cooking"},
    # emotions / bullying
    "bullying": {"bully","anger","shame"},
    # outdoors / play
    "adventure": {"climbing","exploring","parades","floats","candy","roof","porch"},
    "biking": {"bike","biker","training wheels","coasting","downhill","pedal"},
    # money stress
    "money": {"broke","departure tax","couldn’t afford","couldn't afford","lion’s share","lion's share"},
    # housing / living arrangements
    "housing": {"house-sharing","co-rented","mansion","third floor","room of my own"},
    # music
    "music": {"drummer","practicing","drums"},
}

COUNTRY_TITLES = {"US","U.S.","USA","U.S.A.","United States","Ireland","Germany","England","France"}
TITLE_EXCEPTIONS = {"I","We","My"}

AGE_PAT   = re.compile(r"\b(age|aged)\s*(\d{1,2})\b", re.IGNORECASE)
YEAR_PAT  = re.compile(r"\b(19\d{2}|20\d{2})\b")
PROPER_RE = re.compile(r"\b([A-Z][\w\-\’']+)(?:\s+[A-Z][\w\-\’']+){0,2}\b")

# Word-age detection (e.g., "I was nine", "when I was seven")
AGE_WORDS = {
    "one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,
    "eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18
}
AGE_WORD_PAT = re.compile(
    r"\b(?:i\s+was|i\s+turned|when\s+i\s+was)\s+("
    r"one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen"
    r")\b", re.IGNORECASE
)

SYSTEM_ARCHIVIST = "You are a careful archivist for Toddric. Be concise, accurate, and neutral."
SYSTEM_COACH     = "You are Toddric, a practical, kind coach. Use short, concrete steps and avoid therapy/legal advice."

# ------------------ Splitting ------------------
def split_sentences(text: str) -> List[str]:
    text = text.replace("\r\n","\n").replace("\r","\n").strip()
    # Split to lines (blank-line OR single newline), then split lines to sentences
    lines = [ln.strip() for ln in re.split(r"\n\s*\n+|\n", text) if ln.strip()]
    sents = []
    for ln in lines:
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z“"])', ln)
        sents += [p.strip() for p in parts if p.strip()]
    return sents

# ------------------ Metadata ------------------
def find_ages_numeric(s: str) -> list[int]:
    out = []
    for m in AGE_PAT.finditer(s):
        try: out.append(int(m.group(2)))
        except: pass
    return out

def find_ages_words(s: str) -> list[int]:
    return [AGE_WORDS[w.lower()] for w in AGE_WORD_PAT.findall(s) if w.lower() in AGE_WORDS]

def find_ages(s: str) -> list[int]:
    return sorted({*find_ages_numeric(s), *find_ages_words(s)})

def find_years(s: str) -> list[int]:
    return sorted({int(y) for y in YEAR_PAT.findall(s)})

def guess_locations(s: str) -> list[str]:
    s_lc = s.lower()
    hits = set()
    for hint in LOCATION_HINTS:
        if hint in s_lc:
            for m in re.finditer(r"([A-Z][a-zA-Z\-]+(?:,\s*[A-Z][a-zA-Z\-]+)?)", s):
                cand = m.group(1)
                if hint.split(",")[0] in cand.lower():
                    hits.add(cand.strip(", "))
            if not hits:
                hits.add(hint.title())
    for m in re.finditer(r"\b([A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+){0,2})\b", s):
        phrase = m.group(1).strip()
        if phrase in TITLE_EXCEPTIONS: continue
        if phrase.lower() in LOCATION_HINTS: hits.add(phrase)
        if any(ctry in phrase for ctry in COUNTRY_TITLES): hits.add(phrase)
    return sorted(hits)

TITLE_PREFIX = re.compile(r"^(Mr\.|Mrs\.|Ms\.|Dr\.)\s+", re.IGNORECASE)
def _strip_title_plural(s: str) -> str:
    s = TITLE_PREFIX.sub("", s).strip()
    return s[:-1] if s.endswith("s") else s   # "Isbells" -> "Isbell"

def guess_people(s: str) -> list[str]:
    people = set()
    for name in PEOPLE_WHITELIST:
        if re.search(rf"\b{re.escape(name)}\b", s):
            people.add(name)
    for m in PROPER_RE.finditer(s):
        token = _strip_title_plural(m.group(0).strip())
        if token in TITLE_EXCEPTIONS: continue
        if token.lower() in {"another","at","back","in","when","then","soon","finally","not","unfortunately"}: continue
        if token.isupper() and len(token)>3: continue
        if any(ch.isdigit() for ch in token): continue
        if token.lower() in LOCATION_HINTS: continue
        people.add(token)
    return sorted({p for p in people if len(p)>1 and not p.endswith(".")})

def detect_themes(s: str) -> list[str]:
    s_lc = s.lower()
    themes = []
    for theme, words in THEME_KEYWORDS.items():
        if any(w in s_lc for w in words):
            themes.append(theme)
    return sorted(set(themes))

# ------------- Sentence record schema -------------
def make_sentence_record(text: str, idx: int) -> Dict[str, Any]:
    return {
        "source": "memories",
        "version": "v1",
        "memory": {
            "id": str(uuid.uuid4()),
            "seq": idx,
            "text": text,
            "ages": find_ages(text),
            "years": find_years(text),
            "locations": guess_locations(text),
            "people": guess_people(text),
            "themes": detect_themes(text),
        }
    }

# ---------------- Chunking & SFT ----------------
def same_bucket(a: Dict, b: Dict) -> bool:
    am, bm = a.get("memory", {}), b.get("memory", {})
    if set(am.get("ages",[])) & set(bm.get("ages",[])): return True
    if set(am.get("years",[])) & set(bm.get("years",[])): return True
    if set(am.get("locations",[])) & set(bm.get("locations",[])): return True
    return False

def chunk_sentences(rows: List[Dict], min_sent=2, max_sent=6) -> List[List[Dict]]:
    chunks, cur = [], []
    for r in rows:
        if not cur:
            cur=[r]; continue
        if len(cur)>=max_sent:
            chunks.append(cur); cur=[r]; continue
        if same_bucket(cur[-1], r):
            cur.append(r)
        else:
            chunks.append(cur); cur=[r]
    if cur: chunks.append(cur)
    # merge tiny tails
    fixed=[]
    for ch in chunks:
        if len(ch)<min_sent and fixed: fixed[-1].extend(ch)
        else: fixed.append(ch)
    # recut if needed
    out=[]
    for ch in fixed:
        for i in range(0,len(ch),max_sent):
            part=ch[i:i+max_sent]
            if len(part)>=min_sent or not out: out.append(part)
            else: out[-1].extend(part)
    return out

def join_text(chunk: List[Dict]) -> str:
    return " ".join(c["memory"]["text"].strip() for c in chunk if c.get("memory", {}).get("text"))

def merge_meta(chunk: List[Dict]) -> Dict[str, Any]:
    mm = {"ages": set(), "years": set(), "locations": set(), "people": set(), "themes": set()}
    for c in chunk:
        m = c.get("memory", {})
        for k in mm: mm[k].update(m.get(k, []))
    return {k: sorted(v) for k, v in mm.items()}

def sft_item(system: str, user: str, assistant: str = "", tags=None) -> Dict[str, Any]:
    return {
        "system": system,
        "messages": [
            {"role":"user","content": user},
            {"role":"assistant","content": assistant}
        ],
        "tags": tags or []
    }

def build_prompts(chunk_text: str, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    facts_user = (
        "Extract key facts from the memory below. Return JSON with: "
        "summary, who, where, when, notable_events, emotions, consequences.\n\n"
        f"Memory:\n{chunk_text}\n\nMetadata hint: {meta}"
    )
    summary_user = (
        "Summarize this memory in ≤3 sentences, then list 3 lessons as bullets. "
        "Keep it specific to the text, avoid speculation.\n\n"
        f"Memory:\n{chunk_text}"
    )
    advice_user = (
        "Given this memory, produce 3 short pieces of practical advice I can use today. "
        "Keep each to one sentence. Avoid therapy or legal advice.\n\n"
        f"Memory:\n{chunk_text}"
    )
    return [
        sft_item(SYSTEM_ARCHIVIST, facts_user, "", tags=["memories","facts_json","v1"]),
        sft_item(SYSTEM_ARCHIVIST, summary_user, "", tags=["memories","summary_lessons","v1"]),
        sft_item(SYSTEM_COACH,     advice_user, "", tags=["memories","advice_coach","v1"]),
    ]

# ----------------- I/O helpers -----------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def list_txt_files(root: str, pattern: str) -> List[str]:
    out=[]
    for base, _dirs, files in os.walk(root):
        for f in files:
            if f.startswith("."): continue
            if fnmatch.fnmatch(f, pattern):
                out.append(os.path.join(base, f))
    return sorted(out)

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_chunks_jsonl(path: str, chunks: List[List[Dict]], source_name: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for idx, ch in enumerate(chunks, 1):
            text = join_text(ch)
            meta = merge_meta(ch)
            row = {
                "source": "memories",
                "version": "v1",
                "chunk": {
                    "source_file": os.path.basename(source_name),
                    "chunk_id": f"{os.path.splitext(os.path.basename(source_name))[0]}::c{idx}",
                    "seq": idx,
                    "text": text,
                    "meta": meta,
                    "sentence_ids": [c["memory"]["id"] for c in ch],
                    "sentence_seq": [c["memory"]["seq"] for c in ch],
                }
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

# ----------------- Main pipeline -----------------
def process_file(src_path: str, out_root: str, min_sent: int, max_sent: int) -> Dict[str, Any]:
    name = os.path.splitext(os.path.basename(src_path))[0]
    sentences_dir = os.path.join(out_root, "sentences")
    chunks_dir    = os.path.join(out_root, "chunks")
    sft_dir       = os.path.join(out_root, "sft")
    out_sent_path = os.path.join(sentences_dir, f"{name}.jsonl")
    out_chunk_path= os.path.join(chunks_dir,  f"{name}.jsonl")
    out_sft_path  = os.path.join(sft_dir,     f"{name}.jsonl")

    raw = open(src_path, "r", encoding="utf-8-sig").read()
    sents = split_sentences(raw)
    sentence_rows = [make_sentence_record(s, i+1) for i, s in enumerate(sents)]

    # chunk -> prompts
    chunks = chunk_sentences(sentence_rows, min_sent=min_sent, max_sent=max_sent)
    sft_rows = []
    for ch in chunks:
        text = join_text(ch)
        meta = merge_meta(ch)
        sft_rows.extend(build_prompts(text, meta))

    # write (always overwrite)
    write_jsonl(out_sent_path, sentence_rows)
    write_chunks_jsonl(out_chunk_path, chunks, src_path)
    write_jsonl(out_sft_path, sft_rows)

    return {
        "file": src_path,
        "sentences": len(sentence_rows),
        "chunks": len(chunks),
        "sft_items": len(sft_rows),
        "out_sentences": out_sent_path,
        "out_chunks": out_chunk_path,
        "out_sft": out_sft_path,
    }

def main():
    ap = argparse.ArgumentParser(description="Batch TXT -> sentences JSONL -> chunks JSONL -> SFT JSONL.")
    ap.add_argument("input_dir")
    ap.add_argument("out_root")
    ap.add_argument("--pattern", default="*.txt", help="Glob for input files (default: *.txt)")
    ap.add_argument("--min-sent", type=int, default=2, help="Min sentences per chunk (default: 2)")
    ap.add_argument("--max-sent", type=int, default=6, help="Max sentences per chunk (default: 6)")
    args = ap.parse_args()

    ensure_dir(args.out_root)
    ensure_dir(os.path.join(args.out_root, "sentences"))
    ensure_dir(os.path.join(args.out_root, "chunks"))
    ensure_dir(os.path.join(args.out_root, "sft"))

    files = list_txt_files(args.input_dir, args.pattern)
    if not files:
        print("No input files found.", file=sys.stderr)
        sys.exit(2)

    summary = []
    for fp in files:
        try:
            res = process_file(fp, args.out_root, args.min_sent, args.max_sent)
            print(f"[OK ] {os.path.basename(fp)} → sents={res['sentences']} chunks={res['chunks']} sft={res['sft_items']}")
            summary.append(res)
        except Exception as e:
            print(f"[ERR] {fp}: {e}", file=sys.stderr)

    # manifest
    manifest = os.path.join(args.out_root, "manifest.json")
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"✅ Done. Manifest: {manifest}")

if __name__ == "__main__":
    main()
