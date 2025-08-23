from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
tok = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
base = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta", torch_dtype="auto")
model = PeftModel.from_pretrained(base, "./ckpts/toddric-sprint").eval()

SENTINEL=open("sentinel_guardrails.txt").read()
messages=[
  {"role":"system","content":SENTINEL},
  {"role":"user","content":"Summarize this email in 3 bullets:\n\n<PASTE EMAIL TEXT>"},
]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
out = model.generate(**tok(text, return_tensors="pt").to(model.device), max_new_tokens=220)
print(tok.decode(out[0], skip_special_tokens=True))
