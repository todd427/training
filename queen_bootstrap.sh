#!/bin/bash
set -euo pipefail

echo "=== Queen Bootstrap (prep & SSH key) ==="

# -----------------------------
# Ensure directory structure
# -----------------------------
mkdir -p /workspace/training/{ckpts,scripts,repo}
mkdir -p /workspace/release
echo "[*] Ensured /workspace/training structure is present (repo will provide 'out/')."

# -----------------------------
# System packages
# -----------------------------
echo "[*] Installing system packages..."
sudo apt-get update -y
sudo apt-get install -y git tmux htop unzip wget curl jq build-essential vim

# -----------------------------
# Python venv + deps
# -----------------------------
VENV=/workspace/venv312
if [ ! -d "$VENV" ]; then
  echo "[*] Creating venv $VENV"
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --upgrade pip wheel setuptools
pip install -U huggingface_hub \
               "transformers>=4.43" "accelerate>=0.33" "datasets>=2.19" \
               "trl>=0.9.6" "deepspeed>=0.14.0" "peft>=0.11.0"

# -----------------------------
# Hugging Face auth (optional)
# -----------------------------
if command -v hf >/dev/null 2>&1; then
  if ! hf repo list >/dev/null 2>&1; then
    echo "[*] You are not logged into Hugging Face yet. Run 'hf auth login' after bootstrap."
  else
    echo "[*] Hugging Face auth OK."
  fi
fi

# -----------------------------
# SSH key setup
# -----------------------------
if [ ! -f ~/.ssh/id_ed25519 ]; then
  echo "[*] Generating SSH key..."
  mkdir -p ~/.ssh
  ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 -C "runpod-queen"
  chmod 700 ~/.ssh
  chmod 600 ~/.ssh/id_ed25519

  echo
  echo "=== ADD THIS PUBLIC KEY TO GITHUB (https://github.com/settings/keys) ==="
  cat ~/.ssh/id_ed25519.pub
  echo "======================================================================="
  echo "[!] After adding the key to GitHub, run: ./queen_train.sh"
  exit 0
else
  echo "[*] SSH key already exists. Proceed directly to ./queen_train.sh"
fi

