#!/bin/bash
set -euo pipefail

echo "=== Queen Train (clone/pull + run SFT) ==="

# -----------------------------
# Paths & config
# -----------------------------
export HF_HOME=/workspace/.cache/huggingface
VENV=/workspace/venv312

REPO_DIR=/workspace/training/repo
REPO_URL=git@github.com:todd427/training.git

# Training assets expected from repo:
OUT_DIR="/workspace/training/repo/out"               # <- out/ must be in repo
SCRIPT_DIR="/workspace/training/repo/scripts"        # sft_trainer_full.py, ds/fsdp configs
TRAINER="$SCRIPT_DIR/sft_trainer_full.py"

# Checkpoints/output
CKPT_DIR="/workspace/training/ckpts/oss20b-sft"

# Base model
MODEL_ID="openai/oss-gpt-20b"

# Backend: 'ds' (DeepSpeed) or 'fsdp' (PyTorch FSDP)
TRAIN_BACKEND=${TRAIN_BACKEND:-ds}
echo "[*] Selected backend: ${TRAIN_BACKEND}"

# -----------------------------
# Ensure venv is active
# -----------------------------
if [ ! -d "$VENV" ]; then
  echo "[FATAL] Python venv not found at $VENV. Run queen_bootstrap.sh first."
  exit 2
fi
source "$VENV/bin/activate"

# -----------------------------
# GitHub SSH known_hosts
# -----------------------------
mkdir -p ~/.ssh
chmod 700 ~/.ssh
if ! grep -q "github.com" ~/.ssh/known_hosts 2>/dev/null; then
  echo "[*] Adding github.com to known_hosts..."
  ssh-keyscan github.com >> ~/.ssh/known_hosts
fi

# -----------------------------
# Clone or pull repo
# -----------------------------
if [ ! -d "$REPO_DIR/.git" ]; then
  echo "[*] Cloning training repo..."
  git clone "$REPO_URL" "$REPO_DIR"
else
  echo "[*] Pulling latest changes..."
  (cd "$REPO_DIR" && git pull --rebase)
fi

# -----------------------------
# Verify repo-provided OUT_DIR & trainer
# -----------------------------
if [ ! -d "$OUT_DIR" ]; then
  echo "[FATAL] Expected $OUT_DIR from your git repo, but directory not found."
  echo "        Ensure 'out/' is committed to the repo."
  exit 2
fi
if ! ls "$OUT_DIR"/*.jsonl >/dev/null 2>&1; then
  echo "[FATAL] No .jsonl files found in $OUT_DIR. Did you forget to add them to the repo?"
  exit 2
fi
if [ ! -f "$TRAINER" ]; then
  echo "[FATAL] Trainer script not found: $TRAINER"
  exit 2
fi

# -----------------------------
# Quick dry-run model load
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
export NCCL_IB_DISABLE=1          # typical on PCIe pods
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

COMMON_ARGS=(
  --model "$MODEL_ID"
  --train_jsonl \
    "$OUT_DIR/chatgpt.sft.jsonl:1" \
    "$OUT_DIR/epub.sft.jsonl:1" \
    "$OUT_DIR/books.sft.jsonl:1" \
    "$OUT_DIR/sent.sft.jsonl:2" \
    "$OUT_DIR/memories.sft.jsonl:15"
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
    deepspeed --num_gpus "$NUM_GPUS" "$TRAINER" \
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
    torchrun --standalone --nproc_per_node "$NUM_GPUS" "$TRAINER" \
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

