import argparse
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
import os, json

def load_jsonl_dataset(weighted_files, tokenizer, max_seq_len):
    texts = []
    for path, weight in weighted_files:
        with open(path, "r", encoding="utf-8") as fh:
            lines = [line.strip() for line in fh if line.strip()]
            for _ in range(weight):  # oversample by weight
                for line in lines:
                    obj = json.loads(line)
                    text = obj.get("text")
                    if text is None and "messages" in obj:
                        text = "\n".join(m.get("content", "") for m in obj["messages"])
                    if text:
                        texts.append({"text": text})
    print(f"Loaded {len(texts)} samples from {len(weighted_files)} files (after weighting).")
    return Dataset.from_list(texts).map(
        lambda e: tokenizer(e["text"], truncation=True, padding=False, max_length=max_seq_len),
        batched=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--train_jsonl", type=str, nargs="+", required=True,
        help="List of dataset specs, each as file[:weight]. Example: out/books.sft.jsonl:1 out/memories.sft.jsonl:15"
    )

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--fsdp", type=str, default=None)
    parser.add_argument("--fsdp_config", type=str, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Parse file:weight specs
    weighted_files = []
    for spec in args.train_jsonl:
        if ":" in spec:
            path, w = spec.split(":")
            weighted_files.append((path, int(w)))
        else:
            weighted_files.append((spec, 1))

    train_ds = load_jsonl_dataset(weighted_files, tokenizer, args.max_seq_len)


    model = AutoModelForCausalLM.from_pretrained(args.model)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=args.report_to.split(),
        deepspeed=args.deepspeed,
        fsdp=args.fsdp,
        fsdp_config=args.fsdp_config,
        max_seq_length=args.max_seq_len,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        args=training_args,
        packing=args.packing,
    )

    trainer.train()

if __name__ == "__main__":
    main()

