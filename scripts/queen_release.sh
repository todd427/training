#!/bin/bash
set -euo pipefail

VENV=/workspace/venv312
RELEASE_DIR=/workspace/release/oss20b-toddric-sft
CKPT_DIR=/workspace/training/ckpts/oss20b-sft
HF_REPO=toddie314/toddric-20b-sft

source $VENV/bin/activate

echo "[*] Packaging latest checkpoint..."
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
src="/workspace/training/ckpts/oss20b-sft"
dst="/workspace/release/oss20b-toddric-sft"
m=AutoModelForCausalLM.from_pretrained(src, torch_dtype="bfloat16")
t=AutoTokenizer.from_pretrained(src, use_fast=True)
m.save_pretrained(dst, safe_serialization=True)
t.save_pretrained(dst)
print("Saved release at", dst)
PY

echo "[*] Uploading to Hugging Face..."
hf upload $HF_REPO $RELEASE_DIR --repo-type model
hf repo tag create $HF_REPO v0.1 --repo-type model || true

echo "=== Release complete. OK to shut down Queen. ==="

