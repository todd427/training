import os, json, glob, argparse
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=str, default="out",
        help="Directory containing *.sft.jsonl files"
    )
    parser.add_argument(
        "--model", type=str, default="HuggingFaceH4/zephyr-7b-beta",
        help="Tokenizer model to use"
    )
    args = parser.parse_args()

    # Discover all jsonl files in the source dir
    files = glob.glob(os.path.join(args.source, "*.sft.jsonl"))
    if not files:
        print(f"[!] No .sft.jsonl files found in {args.source}")
        return

    print(f"using tokenizer: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    total_tokens = 0
    file_counts = {}

    for path in files:
        c = 0
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text")
                if text is None and "messages" in obj:
                    # handle chat-style
                    text = "\n".join(m.get("content","") for m in obj["messages"])
                if text:
                    c += len(tok.encode(text))
        file_counts[path] = c
        total_tokens += c

    print("=== Token Counts ===")
    for f, c in file_counts.items():
        print(f"{os.path.basename(f)}: {c:,}")
    print(f"\nTOTAL: {total_tokens:,} tokens")

if __name__ == "__main__":
    main()

