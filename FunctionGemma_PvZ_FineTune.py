# -*- coding: utf-8 -*-
"""
FunctionGemma Fine-tune cho PvZ Bot - Google Colab
Theo tài liệu chính thức của Unsloth: https://docs.unsloth.ai/models

Bước 1: Upload file training_data.json lên Colab
Bước 2: Chạy từng cell
"""

# ============================================
# CELL 1: CÀI ĐẶT
# ============================================
!pip install unsloth
!pip install datasets==4.3.0  # Fix lỗi recursion
!pip install --upgrade transformers trl psutil

# ============================================
# CELL 2: LOAD MODEL (Theo docs Unsloth)
# ============================================
from unsloth import FastLanguageModel
import torch

max_seq_length = 4096

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/functiongemma-270m-it",  # Dùng model từ Unsloth
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    load_in_8bit=False,
    load_in_16bit=True,  # 16bit LoRA
    full_finetuning=False,
)

# Thêm LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

print("✓ Đã load model FunctionGemma!")

# ============================================
# CELL 3: ĐỊNH NGHĨA TOOLS CHO PVZ
# ============================================
# Format theo chuẩn FunctionGemma
def collect_sun(x: int, y: int):
    """
    Click to collect sun at pixel position.
    
    Args:
        x: X pixel coordinate of sun
        y: Y pixel coordinate of sun
    
    Returns:
        result: Success message
    """
    return {"result": f"Collected sun at ({x}, {y})"}

def plant_pea_shooter(row: int):
    """
    Plant a pea shooter at specified row.
    
    Args:
        row: Row number (1-5) to plant pea shooter
    
    Returns:
        result: Success message
    """
    return {"result": f"Planted pea shooter at row {row}"}

def do_nothing():
    """
    Wait and do nothing this turn.
    
    Returns:
        result: Wait message
    """
    return {"result": "Waiting..."}

TOOLS = [collect_sun, plant_pea_shooter, do_nothing]

print("✓ Đã định nghĩa tools!")

# ============================================
# CELL 4: LOAD VÀ FORMAT TRAINING DATA
# ============================================
import json
from datasets import Dataset
from transformers.utils import get_json_schema

# Load data
with open("training_data.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

print(f"✓ Đã load {len(raw_data)} samples")

# Tạo tool schemas
TOOL_SCHEMAS = [get_json_schema(f) for f in TOOLS]

def format_sample(sample):
    """Format theo chuẩn FunctionGemma chat template"""
    game_state = sample["game_state"]
    action = sample["action"]
    args = sample["arguments"]
    
    # Tạo messages
    messages = [
        {
            "role": "user",
            "content": f"Game state: {game_state}\nWhat action should I take?"
        },
        {
            "role": "assistant",
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": action,
                    "arguments": args
                }
            }]
        }
    ]
    
    # Apply chat template với tools
    text = tokenizer.apply_chat_template(
        messages,
        tools=TOOL_SCHEMAS,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": text}

# Format tất cả samples
formatted_data = [format_sample(s) for s in raw_data]
dataset = Dataset.from_list(formatted_data)

print(f"✓ Đã format {len(dataset)} samples")
print("\n--- Sample ---")
print(formatted_data[0]["text"][:800])

# ============================================
# CELL 5: TRAINING
# ============================================
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="pvz_output",
    ),
)

print("✓ Bắt đầu training...")
trainer.train()
print("✓ Training xong!")

# ============================================
# CELL 6: TEST MODEL
# ============================================
import re

def extract_tool_calls(text):
    """Parse function calls từ output của FunctionGemma"""
    def cast(v):
        try: return int(v)
        except:
            try: return float(v)
            except: return v.strip("'\"")

    return [{
        "name": name,
        "arguments": {
            k: cast((v1 or v2).strip())
            for k, v1, v2 in re.findall(r"(\w+):(?:<escape>(.*?)<escape>|([^,}]*))", args)
        }
    } for name, args in re.findall(r"<start_function_call>call:(\w+)\{(.*?)\}<end_function_call>", text, re.DOTALL)]

FastLanguageModel.for_inference(model)

def test_bot(game_state):
    messages = [{"role": "user", "content": f"Game state: {game_state}\nWhat action should I take?"}]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tools=TOOL_SCHEMAS,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    
    out = model.generate(
        **inputs.to(model.device),
        max_new_tokens=64,
        top_p=0.95,
        top_k=64,
        temperature=1.0
    )
    
    output = tokenizer.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=False)
    tool_calls = extract_tool_calls(output)
    
    return output, tool_calls

# Test
print("\n" + "="*50)
print("TEST PVZ BOT")
print("="*50)

test_cases = [
    "Sun at (300, 150). No zombie. Can plant: True",
    "No sun. Zombie row [2]. Can plant: True", 
    "No sun. No zombie. Can plant: False",
    "Sun at (450, 200). Zombie row [3]. Can plant: True",
]

for test in test_cases:
    print(f"\n📥 Input: {test}")
    output, calls = test_bot(test)
    print(f"📤 Raw: {output[:100]}...")
    print(f"🎯 Parsed: {calls}")

# ============================================
# CELL 7: SAVE MODEL
# ============================================
model.save_pretrained("pvz_functiongemma_lora")
tokenizer.save_pretrained("pvz_functiongemma_lora")

# Zip để download
!zip -r pvz_functiongemma_lora.zip pvz_functiongemma_lora/

print("\n✓ Đã lưu model!")
print("📥 Download file: pvz_functiongemma_lora.zip")
print("\nSau khi download, giải nén và dùng với pvz_bot.py")
