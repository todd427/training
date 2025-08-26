Here’s a drop-in `README.md` for the Queen scripts. Paste this at the root of your **training** repo (or in `/workspace/training/repo/README.md` after cloning).

---

# Queen (RunPod) — oss-gpt-20B SFT Runbook

This repo contains scripts to **bootstrap** a fresh RunPod, **clone** your training code, and **launch** supervised fine-tuning of **oss-gpt-20B** with either **DeepSpeed ZeRO-3** (default) or **FSDP**.

## Contents

```
/scripts/
  sft_trainer_full.py       # Full SFT trainer used by Queen
  ds_zero3.json             # DeepSpeed ZeRO-3 config
  fsdp_config.json          # FSDP config (NeoX wrap)
  queen_release.sh          # (optional) package & upload to HF

/ (root)
  queen_bootstrap.sh        # Step 1: prep pod + create SSH key (stops)
  queen_train.sh            # Step 2: clone/pull repo + launch training
  out/                      # <-- from git, holds *.sft.jsonl (training data)
  ckpts/                    # runtime output (ignored by git)
  release/                  # runtime output (ignored by git)
```

> Note: We intentionally **do not create** `out/` in scripts. It must come from git and contain your `*.sft.jsonl`.

---

## Quick start (fresh RunPod)

1. Create base folder and run bootstrap:

```bash
mkdir -p /workspace/training && cd /workspace/training
# add queen_bootstrap.sh to this folder, then:
chmod +x queen_bootstrap.sh
./queen_bootstrap.sh
```

* Installs: `git`, `vim`, Python venv, `transformers`, `accelerate`, `deepspeed`, `trl`, `peft`, `huggingface_hub`.
* If no SSH key exists, it **generates** one and prints the **public key**.

2. Add the printed public key to GitHub:

* Open [https://github.com/settings/keys](https://github.com/settings/keys) → **New SSH key** → paste the key → save.

3. Clone+train:

```bash
# add queen_train.sh, then:
chmod +x queen_train.sh
TRAIN_BACKEND=ds ./queen_train.sh | tee queen_run.log     # DeepSpeed (default)
# or
TRAIN_BACKEND=fsdp ./queen_train.sh | tee queen_run.log   # FSDP
```

---

## Requirements

* RunPod PyTorch image (CUDA 12.x, Python 3.10+).
* GPUs: 4× H100 80GB (PCIe) recommended.
* Disk: ≥ 1 TB persistent.
* Internet egress to pull models/datasets and push to Hugging Face.

---

## Hugging Face (optional but recommended)

Login any time (before release/upload steps):

```bash
hf auth login
```

We set `HF_HOME=/workspace/.cache/huggingface` so caches persist across runs.

---

## Backends

* **DeepSpeed ZeRO-3** (recommended default)

  * Config: `scripts/ds_zero3.json`
  * Launch: `TRAIN_BACKEND=ds ./queen_train.sh`

* **FSDP (PyTorch)**

  * Config: `scripts/fsdp_config.json` (auto-wrap `GPTNeoXLayer`)
  * Launch: `TRAIN_BACKEND=fsdp ./queen_train.sh`

---

## Trainer (full SFT)

The launcher runs `scripts/sft_trainer_full.py` with these defaults:

* `--model openai/oss-gpt-20b`
* `--epochs 2`
* `--learning_rate 2e-5`
* `--batch_size 1`
* `--grad_accum 128`
* `--max_seq_len 2048`
* `--bf16 --packing --gradient_checkpointing`
* `--save_steps 500 --eval_steps 500 --logging_steps 10`

Training JSONL files expected at:

```
repo/out/chatgpt.sft.jsonl
repo/out/epub.sft.jsonl
repo/out/memories.sft.jsonl
repo/out/sent.sft.jsonl
repo/out/books.sft.jsonl   # optional
```

> Edit paths/flags inside `queen_train.sh` if your dataset mix changes.

---

## Release & Upload (optional)

After training:

```bash
# Packages the latest checkpoint and uploads to HF repo (edit repo name inside the script)
chmod +x scripts/queen_release.sh
./scripts/queen_release.sh
```

This saves a clean model into `/workspace/release/oss20b-toddric-sft/`, runs:

```
hf upload <your-user>/toddric-20b-sft /workspace/release/oss20b-toddric-sft --repo-type model
hf repo tag create <your-user>/toddric-20b-sft v0.1 --repo-type model
```

---

## Cost-control checklist

* Use **tmux** so training continues if SSH drops:

  ```bash
  tmux new -s queen
  ./queen_train.sh | tee queen_run.log
  ```
* Do a **100-step benchmark** once if you’re unsure about step time:

  * Add flags: `--max_steps 100 --evaluation_strategy "no" --save_steps 0`
* Keep checkpoints/evals **infrequent** during the main run.
* Shut down the pod **immediately** after upload/tag. Do docs locally.

---

## Troubleshooting

* **Repo not found / no `out/*.jsonl`**
  Ensure `out/` is committed to git. `queen_train.sh` fails fast if not found.

* **Git auth fails**
  Run bootstrap → add the printed SSH pubkey at GitHub → re-run train.

* **HF upload metadata error**
  In your model card YAML front-matter:
  `base_model:` must be a valid HF model id (e.g., `HuggingFaceH4/zephyr-7b-beta`), not a path.

* **OOM**
  Increase `--grad_accum`, reduce `--max_seq_len` (e.g., 1536/1024), keep gradient checkpointing on.

* **Comms stalls**
  Env vars set by the scripts:

  ```
  NCCL_DEBUG=warn
  NCCL_IB_DISABLE=1
  NCCL_P2P_DISABLE=0
  NCCL_SOCKET_IFNAME=eth0
  TOKENIZERS_PARALLELISM=false
  ```

  Confirm GPU utilization with `watch -n 5 nvidia-smi`.

---

## .gitignore (recommended)

```
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
*.so

# Virtualenvs
.venv/
venv*/
env*/

# Checkpoints / releases
ckpts/
release/
*.safetensors
*.bin
*.pt
*.pth
*.tar.gz

# Caches & logs
.cache/
.huggingface/
wandb/
logs/
*.log
*.out
*.err

# OS
.DS_Store
Thumbs.db
```

---

**Workflow summary**

1. `queen_bootstrap.sh` → installs deps + prints SSH pubkey → add to GitHub.
2. `queen_train.sh` → clones/pulls repo → verifies `out/*.jsonl` → dry-run → starts SFT.
3. (Optional) `scripts/queen_release.sh` → package → HF upload → tag → shut down pod.
