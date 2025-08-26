#!/bin/bash
set -euo pipefail

echo "=== Queen Bootstrap (prep & SSH key) ==="

# -----------------------------
# Basic dirs (adjust paths if you mounted a volume)
# -----------------------------
mkdir -p /workspace/training/{ckpts,scripts,repo} /workspace/release
echo "[*] Ensured /workspace/training structure."

# -----------------------------
# Optional system packages (skip if your image already has them)
# -----------------------------
if command -v apt-get >/dev/null 2>&1; then
  echo "[*] Installing system packages (non-interactive)..."
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y || true
  apt-get install -y --no-install-recommends git tmux htop unzip wget curl jq build-essential vim || true
fi

# -----------------------------
# Python venv + deps
# -----------------------------
VENV=/workspace/venv312
if [ ! -d "$VENV" ]; then
  echo "[*] Creating venv $VENV"
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# Core ML stack (pin minimally; you can refine later)
python -m pip install -U \
  huggingface_hub \
  "transformers>=4.55" "accelerate>=0.33" "datasets>=2.19" \
  "trl>=0.21.0" "peft>=0.11.0" "bitsandbytes>=0.43"

# -----------------------------
# Hugging Face auth (optional)
# -----------------------------
if command -v huggingface-cli >/dev/null 2>&1; then
  if ! huggingface-cli whoami >/dev/null 2>&1; then
    echo "[*] Not logged into Hugging Face (that's ok if you use open models)."
    echo "    Run: huggingface-cli login   (or set HUGGINGFACE_HUB_TOKEN env)"
  else
    echo "[*] Hugging Face auth OK."
  fi
fi

# -----------------------------
# SSH key setup
# -----------------------------
mkdir -p ~/.ssh && chmod 700 ~/.ssh
if [ ! -f ~/.ssh/id_ed25519 ]; then
  echo "[*] Generating SSH key..."
  ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 -C "runpod-queen"
  chmod 600 ~/.ssh/id_ed25519

  # Trust GitHub host key to avoid interactive prompt
  ssh-keyscan -t rsa,ecdsa,ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null || true
  chmod 644 ~/.ssh/known_hosts

  # Helpful SSH config
  cat > ~/.ssh/config <<'CFG'
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
  StrictHostKeyChecking accept-new
  ServerAliveInterval 30
CFG
  chmod 600 ~/.ssh/config

  echo
  echo "=== ADD THIS PUBLIC KEY TO GITHUB (https://github.com/settings/keys) ==="
  cat ~/.ssh/id_ed25519.pub
  echo "======================================================================="
  echo "[!] After adding the key to GitHub, just restart the pod or run the clone command below."
  # Keep the container alive so you can copy the key & add to GitHub
  tail -f /dev/null
fi

# Ensure GitHub host key exists even if key was pre-existing
ssh-keyscan -t rsa,ecdsa,ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null || true

# -----------------------------
# Repo bootstrap (clone or pull)
# -----------------------------
cd /workspace
if [ ! -d training/.git ]; then
  echo "[*] Cloning repo..."
  GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=accept-new" git clone git@github.com:todd427/training.git
else
  echo "[*] Updating repo..."
  cd training
  git config --global user.name "Todd J. McCaffrey"
  git config --global user.email "todd@toddwriter.com"
  git stash push -u -m "pre-start pull" || true
  GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=accept-new" git pull --rebase || true
  git stash pop || true
fi

# -----------------------------
# Ready for manual or auto run
# -----------------------------
echo "[*] Bootstrap complete. You can now run training, e.g.:"
echo "    source $VENV/bin/activate && cd /workspace/training && \\"
echo "    python scripts/run_sft.py --model Qwen/Qwen2.5-3B-Instruct \\"
echo "      --out ckpts/toddric-3b-lora-v0 \\"
echo "      --train_jsonl out/sent.sft.jsonl out/memories.sft.jsonl \\"
echo "      --epochs 1 --lr 2e-4 --batch_size 1 --grad_accum 16 --max_seq_len 2048"

# Keep container alive for interactive use; replace with your command to auto-run
tail -f /dev/null
