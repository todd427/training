# eval/smoke_qwen_adapter_hardened.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch, time, json, os

BASE = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER = "./ckpts/toddric-3b-lora-v0"
OUTDIR = "eval/run_20250826"; os.makedirs(OUTDIR, exist_ok=True)

# Prefer bf16 on 4060 if available, else fp16
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token  # important for stable generate

model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=dtype, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

# Helper: robust chat build
def build_ids(messages):
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True  # ensure assistant slot is opened
    )
    return tok(prompt, return_tensors="pt").to(model.device)

# Identify Qwen's end token (usually <|im_end|>)
im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
bad_words = None
if im_end_id is not None:
    # Prevent immediate stop on the very first token
    bad_words = [[im_end_id]]

gen_kwargs = dict(
    max_new_tokens=192,
    min_new_tokens=16,       # <-- ensures we don't stop at 0 tokens
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.05,
    pad_token_id=tok.pad_token_id,
    eos_token_id=tok.eos_token_id,
)
if bad_words:
    gen_kwargs["bad_words_ids"] = bad_words

tests = [
  {"role":"user","content":"Give me a 2-tweet announcement for Toddric’s alpha in my voice."},
  {"role":"user","content":"Briefly: how to mount a CIFS share on Ubuntu; include the exact command."},
  {"role":"user","content":"Summarize the L.A. Witch series tone in 1 paragraph."},
]

outs = []
for m in tests:
    messages = [
        {"role":"system","content":"You are Toddric, Todd’s helpful assistant with Todd’s tone."},
        m
    ]
    ids = build_ids(messages)
    t0 = time.time()
    out = model.generate(**ids, **gen_kwargs)
    text = tok.batch_decode(out, skip_special_tokens=True)[0]
    outs.append({
        "messages": messages,
        "output": text.strip(),
        "latency_s": round(time.time()-t0, 2),
        "gen_kwargs": {k: v for k,v in gen_kwargs.items() if k != "bad_words_ids"}
    })

with open(f"{OUTDIR}/smoke_adapter_qwen_hardened.jsonl","w",encoding="utf-8") as f:
    for o in outs: f.write(json.dumps(o, ensure_ascii=False)+"\n")
print("Saved", f"{OUTDIR}/smoke_adapter_qwen_hardened.jsonl")
