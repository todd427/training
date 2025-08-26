#!/usr/bin/env python
import gradio as gr, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER = "ckpts/toddric-3b-lora-v0"

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained(BASE, use_fast=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(BASE, trust_remote_code=True, device_map="auto", quantization_config=bnb)
model = PeftModel.from_pretrained(model, ADAPTER)

def respond(messages, max_new_tokens=384, temperature=0.7, top_p=0.95):
    # messages is [["user",".."],["assistant",".."],...]
    chat = [{"role": "system", "content": "You are Toddric, concise and helpful."}]
    for role, content in messages:
        chat.append({"role": role, "content": content})
    prompt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_p=top_p)
    text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return text

with gr.Blocks() as demo:
    gr.Markdown("# Toddric 3B (LoRA) — Local Chat")
    chatbot = gr.Chatbot(height=520)
    with gr.Row():
        msg = gr.Textbox(scale=4, placeholder="Ask me anything…")
        send = gr.Button("Send", variant="primary")
    with gr.Accordion("Generation Settings", open=False):
        mx = gr.Slider(64, 2048, 384, step=64, label="max_new_tokens")
        temp = gr.Slider(0.0, 1.5, 0.7, step=0.05, label="temperature")
        topp = gr.Slider(0.1, 1.0, 0.95, step=0.05, label="top_p")

    def user_submit(user_message, chat_history):
        if not user_message:
            return gr.update(), chat_history
        chat_history = chat_history + [[user_message, None]]
        return "", chat_history

    def bot_reply(chat_history, mx, temp, topp):
        response = respond([[("user" if i%2==0 else "assistant")[0:4], m] for i, (m, _) in enumerate(chat_history) if m],
                           max_new_tokens=mx, temperature=temp, top_p=topp)
        chat_history[-1][1] = response
        return chat_history

    send.click(user_submit, [msg, chatbot], [msg, chatbot]).then(
        bot_reply, [chatbot, mx, temp, topp], [chatbot]
    )
    msg.submit(user_submit, [msg, chatbot], [msg, chatbot]).then(
        bot_reply, [chatbot, mx, temp, topp], [chatbot]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
