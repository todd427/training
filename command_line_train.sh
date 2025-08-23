python sft_train.py \
  --model HuggingFaceH4/zephyr-7b-beta \
  --train_jsonl out/sent.sft.jsonl out/memories.sft.jsonl out/epub.sft.jsonl \
  --train_weights 0.5 0.25 0.25 \
  --output_dir ./ckpts/toddric-zephyr-lora \
  --epochs 3 --lr 1.5e-5 --batch_size 2 --grad_accum 32 \
  --max_seq_len 3072 --bf16 --packing \
  --gradient_checkpointing --use_flash_attn
