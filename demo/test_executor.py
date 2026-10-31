# -*- coding: utf-8 -*-
"""单独验证 Executor：Qwen3-4B 4-bit 加载 + enable_thinking=False 的 chat template 是否可用 + 显存占用。"""
import os
import sys
import torch

sys.path.insert(0, r"D:\PathCode")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def main():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"VRAM 初始 free={free/1e9:.2f}GB / total={total/1e9:.2f}GB")

    ckpt = r"D:\PathCode\checkpoints\Qwen3-4B"
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    print("加载 Qwen3-4B (4-bit)...")
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, quantization_config=bnb, device_map="auto"
    )
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"加载后 VRAM free={free/1e9:.2f}GB (占用约 {(total-free)/1e9:.2f}GB)")

    msgs = [
        {"role": "system", "content": "You are a pathology assistant."},
        {"role": "user", "content": "Reply with only JSON: {\"answer\": \"4\"}"},
    ]
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    print("chat template 前 80 字符:", repr(text[:80]))

    with torch.no_grad():
        ids = tokenizer([text], return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=32)
    ans = tokenizer.decode(out[0][len(ids.input_ids[0]):], skip_special_tokens=True)
    print("生成结果:", repr(ans.strip()))
    print("Executor 验证通过")


if __name__ == "__main__":
    main()
