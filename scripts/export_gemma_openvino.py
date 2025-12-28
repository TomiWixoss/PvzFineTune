# -*- coding: utf-8 -*-
"""
Export FunctionGemma sang OpenVINO để inference nhanh hơn

Usage:
    python scripts/export_gemma_openvino.py
"""

from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

MODEL_PATH = "models/gemma"
OUTPUT_PATH = "models/gemma_openvino"

print(f"Loading model from {MODEL_PATH}...")
model = OVModelForCausalLM.from_pretrained(
    MODEL_PATH,
    export=True,
    load_in_8bit=False
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print(f"Saving to {OUTPUT_PATH}...")
model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)

print("✓ Done! Use models/gemma_openvino for faster inference")
