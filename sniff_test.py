from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta", use_fast=True)
model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta", torch_dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(model, "toddie314/toddric-zephyr7b-lora")

prompts = [
    "Explain quantum physics in the style of Toddric.",
    "Write a short Toddric-style bedtime story about dragons and AI.",
    "Give thoughtful advice to a student nervous about exams."
]

for p in prompts:
    print("Prompt:", p)
    inputs = tok(p, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7, top_p=0.9)
    print(tok.decode(out[0], skip_special_tokens=True))
    print("="*80)
