# -*- coding: utf-8 -*-
"""单独验证 Perceptor 替换：Qwen2.5-VL-3B 4-bit 加载 + 对一张 patch 生成描述。
    排除 VL 量化/processor/chat-template 的坑，再跑完整 demo。"""
import os
import sys
import torch

sys.path.insert(0, r"D:\PathCode")

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from models.inference import patho_r1_describe
from data_processing.utils import load_image

def main():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"VRAM 初始 free={free/1e9:.2f}GB / total={total/1e9:.2f}GB")

    ckpt = r"D:\PathCode\checkpoints\Qwen2.5-VL-3B-Instruct"
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    print("加载 Qwen2.5-VL-3B (4-bit)...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        ckpt, quantization_config=bnb, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(ckpt)
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"加载后 VRAM free={free/1e9:.2f}GB (占用约 {(total-free)/1e9:.2f}GB)")

    img = load_image(r"D:\PathCode\demo\patches\DX1_SICAP001\6803_59786.png")
    print("生成描述中...")
    out = patho_r1_describe(
        img,
        question="What is the primary Gleason pattern of this prostate tissue?",
        patho_r1_processor=processor,
        patho_r1_model=model,
        coords=(6803, 59786),
        magnification=5,
        choices=["3", "4", "5"],
    )
    print("=" * 60)
    print("Perceptor 输出:")
    print(out)
    print("=" * 60)
    print("Perceptor 验证通过 ✓")


if __name__ == "__main__":
    main()
