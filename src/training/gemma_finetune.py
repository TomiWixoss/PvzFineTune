# -*- coding: utf-8 -*-
"""
FunctionGemma Fine-tuning for PvZ Bot
Based on: https://ai.google.dev/gemma/docs/functiongemma/finetuning-with-functiongemma

Run this in Google Colab with GPU
"""

# ============================================
# CELL 1: Install (Restart runtime after!)
# ============================================
# !pip install torch tensorboard
# !pip install transformers datasets accelerate evaluate trl protobuf sentencepiece

# ============================================
# CELL 2: Login to HuggingFace
# ============================================
# from huggingface_hub import login
# login()

# ============================================
# CELL 3: Load Model
# ============================================
"""
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import get_json_schema

base_model = "google/functiongemma-270m-it"

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float32,
    device_map="auto",
    attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained(base_model)

print(f"✓ Model loaded! Device: {model.device}, Dtype: {model.dtype}")
"""

# ============================================
# CELL 4: Define Tools
# ============================================
"""
def collect_sun(x: int, y: int) -> str:
    \"\"\"Click to collect sun at pixel position.\"\"\"
    return "Collected"

def plant_pea_shooter() -> str:
    \"\"\"Plant a pea shooter at row 3.\"\"\"
    return "Planted"

def do_nothing() -> str:
    \"\"\"Wait and do nothing this turn.\"\"\"
    return "Waiting"

TOOLS = [
    get_json_schema(collect_sun), 
    get_json_schema(plant_pea_shooter), 
    get_json_schema(do_nothing)
]
print("✓ Tools defined!")
print(json.dumps(TOOLS, indent=2))
"""

# ============================================
# CELL 5: Load and Format Data
# ============================================
"""
from datasets import Dataset

with open("training_data.json", "r") as f:
    raw_data = json.load(f)

print(f"✓ Loaded {len(raw_data)} samples")

DEFAULT_SYSTEM_MSG = "You are a PvZ game bot. Choose ONE action based on game state."

def create_conversation(sample):
    action = sample["action"]
    args = sample["arguments"]
    
    if action == "collect_sun":
        tool_call = {"type": "function", "function": {"name": "collect_sun", "arguments": {"x": args["x"], "y": args["y"]}}}
    elif action == "plant_pea_shooter":
        tool_call = {"type": "function", "function": {"name": "plant_pea_shooter", "arguments": {}}}
    else:
        tool_call = {"type": "function", "function": {"name": "do_nothing", "arguments": {}}}
    
    return {
        "messages": [
            {"role": "developer", "content": DEFAULT_SYSTEM_MSG},
            {"role": "user", "content": sample["game_state"]},
            {"role": "assistant", "tool_calls": [tool_call]},
        ],
        "tools": TOOLS
    }

dataset = Dataset.from_list(raw_data)
dataset = dataset.map(create_conversation, remove_columns=dataset.features, batched=False)
dataset = dataset.train_test_split(test_size=0.2, shuffle=True)

print(f"✓ Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")
"""

# ============================================
# CELL 6: Training
# ============================================
"""
from trl import SFTTrainer, SFTConfig

args = SFTConfig(
    output_dir="pvz_functiongemma",
    max_length=512,
    packing=False,
    num_train_epochs=10,
    per_device_train_batch_size=4,
    gradient_checkpointing=False,
    optim="adamw_torch",
    logging_steps=10,
    eval_strategy="epoch",
    learning_rate=5e-5,
    fp16=False,
    bf16=False,
    lr_scheduler_type="constant",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    processing_class=tokenizer,
)

print("✓ Starting training...")
trainer.train()
print("✓ Training done!")
"""

# ============================================
# CELL 7: Test
# ============================================
"""
def test_bot(game_state):
    messages = [
        {"role": "developer", "content": DEFAULT_SYSTEM_MSG},
        {"role": "user", "content": game_state},
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages, 
        tools=TOOLS, 
        add_generation_prompt=True, 
        return_dict=True, 
        return_tensors="pt"
    )
    
    out = model.generate(
        **inputs.to(model.device), 
        pad_token_id=tokenizer.eos_token_id, 
        max_new_tokens=64
    )
    output = tokenizer.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=False)
    return output

print("\\n" + "="*50)
print("TEST PVZ BOT")
print("="*50)

test_cases = [
    "HAS_SUN x=300 y=150. NO_ZOMBIE. CAN_PLANT",
    "NO_SUN. HAS_ZOMBIE. CAN_PLANT",
    "NO_SUN. NO_ZOMBIE. CANNOT_PLANT",
]

for t in test_cases:
    print(f"\\n📥 Input: {t}")
    print(f"📤 Output: {test_bot(t)}")
"""

# ============================================
# CELL 8: Save
# ============================================
"""
model.save_pretrained("pvz_functiongemma_final")
tokenizer.save_pretrained("pvz_functiongemma_final")
!zip -r pvz_functiongemma_final.zip pvz_functiongemma_final/
print("\\n✓ Saved! Download pvz_functiongemma_final.zip")
"""

if __name__ == "__main__":
    print("This script is designed for Google Colab")
    print("Run the cells in Colab with GPU for training")
    print("\nAfter training, copy the model to: models/gemma/pvz_functiongemma_final/")
