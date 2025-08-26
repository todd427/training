#!/bin/bash
set -euo pipefail

echo "=== Queen Startup (oss-gpt-20B SFT) ==="

# -----------------------------
# Ensure directory structure
# -----------------------------
mkdir -p /workspace/training/{ckpts,scripts, repo}
mkdir -p /workspace/release
echo "[*] Ensured /workspace/training structure is present."

# -----------------------------
# Config
# -----------------------------
VENV=/workspace/venv312
MODEL_ID="openai/oss-gpt-20b"

OUT_DIR="/workspace/training/out"
CKPT_DIR="/workspace/training/ckpts/oss20b-sft"
SCRIPT_DIR="/workspace/training/scripts"

# Backend: 'ds' (DeepSpeed) or 'fsdp' (PyTorch FSDP)
TRAIN_BACKEND=${TRAIN_BACKEND:-ds}
echo "[*] Selected backend: ${TRAIN_BACKEND}"

# Ensure HF cache is on persistent storage
export HF_HOME=/workspace/.cache/huggingface

# -----------------------------
# System prep
# -----------------------------
echo "[*] Installing system packages..."
sudo apt-get update -y
sudo apt-get install -y git tmux htop unzip wget curl jq build-essential

# -----------------------------
# Python venv
# -----------------------------
if [ ! -d "$VENV" ]; then
  echo "[*] Creating venv $VENV"
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --upgrade pip wheel setuptools

# -----------------------------
# Python deps
# -----------------------------
echo "[*] Installing Python deps..."
pip install -U huggingface_hub \
               "transformers>=4.43" "accelerate>=0.33" "datasets>=2.19" \
               "trl>=0.9.6" "deepspeed>=0.14.0" "peft>=0.11.0"

# -----------------------------
# Hugging Face auth check
# -----------------------------
if ! command -v hf >/dev/null 2>&1; then
  echo "[!] 'hf' CLI not found after install; check environment."
else
  if ! hf repo list >/dev/null 2>&1; then
    echo "[*] Hugging Face not authenticated. Run 'hf auth login' in another shell if needed."
  else
    echo "[*] Hugging Face auth OK."
  fi
fi

# -----------------------------
# Dry-run: quick model load
# -----------------------------
echo "[*] Running dry-run model load..."
python - <<'PY'
import torch
print("GPUs:", torch.cuda.device_count())
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("openai/oss-gpt-20b", torch_dtype="bfloat16", device_map="auto")
t = AutoTokenizer.from_pretrained("openai/oss-gpt-20b", use_fast=True)
print("Dry-run successful.")
PY

# -----------------------------
# Env for NCCL / tokenizers
# -----------------------------
export NCCL_DEBUG=warn
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_BLOCKING_WAIT=0

# -----------------------------
# Launch training
# -----------------------------
echo "[*] Starting SFT training..."
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "[*] Detected GPUs: $NUM_GPUS"

# Common args (shared by both backends)
COMMON_ARGS=(
  --model "$MODEL_ID"
  --train_jsonl "$OUT_DIR/chatgpt.sft.jsonl" "$OUT_DIR/epub.sft.jsonl" \
                "$OUT_DIR/memories.sft.jsonl" "$OUT_DIR/sent.sft.jsonl" \
                "$OUT_DIR/books.sft.jsonl"
  --output_dir "$CKPT_DIR"
  --epochs 2
  --learning_rate 2e-5
  --batch_size 1
  --grad_accum 128
  --warmup_ratio 0.03
  --max_seq_len 2048
  --bf16
  --packing
  --gradient_checkpointing
  --logging_steps 10
  --eval_steps 500
  --save_steps 500
  --report_to none
)

case "$TRAIN_BACKEND" in
  ds|deepspeed)
    DS_CFG="$SCRIPT_DIR/ds_zero3.json"
    if [ ! -f "$DS_CFG" ]; then
      echo "[FATAL] DeepSpeed config not found: $DS_CFG"
      exit 2
    fi
    echo "[*] Launching DeepSpeed ZeRO-3..."
    deepspeed --num_gpus "$NUM_GPUS" sft_train.py \
      "${COMMON_ARGS[@]}" \
      --deepspeed "$DS_CFG"
    ;;
  fsdp)
    FSDP_CFG="$SCRIPT_DIR/fsdp_config.json"
    if [ ! -f "$FSDP_CFG" ]; then
      echo "[FATAL] FSDP config not found: $FSDP_CFG"
      exit 2
    fi
    echo "[*] Launching PyTorch FSDP..."
    torchrun --standalone --nproc_per_node "$NUM_GPUS" sft_train.py \
      "${COMMON_ARGS[@]}" \
      --fsdp "full_shard auto_wrap" \
      --fsdp_config "$FSDP_CFG"
    ;;
  *)
    echo "[FATAL] Unknown TRAIN_BACKEND='$TRAIN_BACKEND' (use 'ds' or 'fsdp')"
    exit 2
    ;;
esac

echo "=== Training started. Monitor with: watch -n 5 nvidia-smi ==="

