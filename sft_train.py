#!/usr/bin/env python3
import argparse, inspect, math, os, torch
from typing import Dict, Any

from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer


# ---------------------------
# Arg parsing
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    # I/O + model
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument(
        "--train_jsonl", nargs="+", required=True,
        help="One or more JSONL files; each row must have a `messages` list"
    )
    ap.add_argument("--output_dir", required=True)

    # Training schedule
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--max_steps", type=int, default=-1)

    # Precision / perf
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--packing", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")

    # Logging / saving / eval
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--eval_steps", type=int, default=500)
    ap.add_argument(
        "--eval_holdout", type=float, default=0.0,
        help="0.0 disables eval; otherwise split this fraction from train"
    )
    ap.add_argument("--report_to", default="none")  # e.g. none|wandb|tensorboard
    return ap.parse_args()


def _supports_arg(cls, arg_name: str) -> bool:
    try:
        sig = inspect.signature(cls.__init__)
        return arg_name in sig.parameters
    except Exception:
        return False


def build_training_args(args) -> TrainingArguments:
    kwargs: Dict[str, Any] = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        report_to=args.report_to,
    )

    # Optional / version-dependent flags
    if _supports_arg(TrainingArguments, "save_total_limit"):
        kwargs["save_total_limit"] = 2
    if _supports_arg(TrainingArguments, "max_steps") and args.max_steps and args.max_steps > 0:
        kwargs["max_steps"] = args.max_steps
    if _supports_arg(TrainingArguments, "bf16"):
        kwargs["bf16"] = bool(args.bf16)
    if _supports_arg(TrainingArguments, "fp16"):
        kwargs["fp16"] = bool(args.fp16)
    if _supports_arg(TrainingArguments, "tf32"):
        kwargs["tf32"] = bool(args.tf32)
    if _supports_arg(TrainingArguments, "eval_steps") and args.eval_steps:
        kwargs["eval_steps"] = args.eval_steps

    # Eval toggles differ by version
    if _supports_arg(TrainingArguments, "evaluation_strategy"):
        kwargs["evaluation_strategy"] = "steps" if args.eval_holdout and args.eval_holdout > 0 else "no"
    elif _supports_arg(TrainingArguments, "do_eval"):
        kwargs["do_eval"] = bool(args.eval_holdout and args.eval_holdout > 0)

    # Gradient checkpointing (moved around across versions)
    if _supports_arg(TrainingArguments, "gradient_checkpointing"):
        kwargs["gradient_checkpointing"] = bool(args.gradient_checkpointing)

    return TrainingArguments(**kwargs)


def main():
    args = parse_args()

    # Perf knobs
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    if args.tf32:
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # Tokenizer (legacy=True keeps old token splitting; avoids behavior shift)
    tok = AutoTokenizer.from_pretrained(args.model, legacy=True)
    # Fallback chat template if none is provided by the tokenizer
    if not getattr(tok, "chat_template", None):
        tok.chat_template = """{{ bos_token if bos_token is defined else '' }}{%- for m in messages -%}
{%- if m['role'] == 'system' -%}<|system|>
{{ m['content'] }}
{%- elif m['role'] == 'user' -%}<|user|>
{{ m['content'] }}
{%- elif m['role'] == 'assistant' -%}<|assistant|>
{{ m['content'] }}
{%- endif -%}
{%- endfor -%}"""

    # Load & combine datasets
    splits = [load_dataset("json", data_files=f, split="train") for f in args.train_jsonl]
    train_all = concatenate_datasets(splits)

    # Optional eval holdout
    eval_ds = None
    if args.eval_holdout and args.eval_holdout > 0:
        train_all = train_all.shuffle(seed=args.seed)
        n_total = len(train_all)
        n_eval = max(1, int(math.floor(n_total * float(args.eval_holdout))))
        eval_ds = train_all.select(range(n_eval))
        train_all = train_all.select(range(n_eval, n_total))

    # Map: messages -> text
    def format_row(ex):
        msgs = ex["messages"]
        try:
            text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            # ultra-safe fallback
            parts = []
            for m in msgs:
                parts.append(f"<|{m['role']}|>\n{m['content']}\n")
            text = "".join(parts)
        return {"text": text}

    train_fmt = train_all.map(
        format_row, remove_columns=train_all.column_names, desc="format train"
    )
    if eval_ds is not None:
        eval_fmt = eval_ds.map(
            format_row, remove_columns=eval_ds.column_names, desc="format eval"
        )
    else:
        eval_fmt = None

    # Dtype selection
    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    # Load model
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype, low_cpu_mem_usage=True, device_map="auto")

    # Training args (version-flex)
    training_args = build_training_args(args)

    # Data collator
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # ----- Build SFTTrainer with version compatibility -----
    sft_sig = inspect.signature(SFTTrainer.__init__).parameters
    def sft_has(param: str) -> bool:
        return param in sft_sig

    sft_kwargs = {
        "model": model,
        "train_dataset": train_fmt,
        "args": training_args,
        "data_collator": collator,
    }
    # Optional / version-dependent SFT args
    if eval_fmt is not None and sft_has("eval_dataset"):
        sft_kwargs["eval_dataset"] = eval_fmt
    if sft_has("tokenizer"):
        sft_kwargs["tokenizer"] = tok
    if sft_has("dataset_text_field"):
        sft_kwargs["dataset_text_field"] = "text"
    if sft_has("max_seq_length"):
        sft_kwargs["max_seq_length"] = args.max_seq_len
    if sft_has("packing"):
        sft_kwargs["packing"] = bool(args.packing)

    trainer = SFTTrainer(**sft_kwargs)

    # Very old TRL: attach tokenizer after init
    if not sft_has("tokenizer"):
        trainer.tokenizer = tok

    # Train & save
    trainer.train()
    try:
        trainer.save_model(args.output_dir)
    except Exception:
        # Some TRL builds save via model.save_pretrained
        model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

