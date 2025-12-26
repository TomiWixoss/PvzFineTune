# -*- coding: utf-8 -*-
"""
PvZ Bot - Inference với FunctionGemma đã fine-tune
"""

import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import get_json_schema

# ============================================
# LOAD MODEL
# ============================================
MODEL_PATH = "pvz_functiongemma_final"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"✓ Model loaded on {device}")

# ============================================
# TOOLS DEFINITION
# ============================================
def collect_sun(x: int, y: int) -> str:
    """Click to collect sun at pixel position."""
    return "Collected"

def plant_pea_shooter() -> str:
    """Plant a pea shooter at row 3."""
    return "Planted"

def do_nothing() -> str:
    """Wait and do nothing."""
    return "Waiting"

TOOLS = [get_json_schema(collect_sun), get_json_schema(plant_pea_shooter), get_json_schema(do_nothing)]
SYSTEM_MSG = "You are a PvZ game bot. Choose ONE action based on game state."

# ============================================
# INFERENCE
# ============================================
def parse_function_call(output):
    """Parse function call từ output"""
    match = re.search(r"call:(\w+)\{([^}]*)\}", output)
    if not match:
        return None, {}
    
    name = match.group(1)
    args_str = match.group(2)
    
    args = {}
    if args_str:
        for pair in args_str.split(","):
            if ":" in pair:
                k, v = pair.split(":", 1)
                try:
                    args[k.strip()] = int(v.strip())
                except:
                    args[k.strip()] = v.strip()
    
    return name, args

def get_action(game_state):
    """Lấy action từ game state"""
    messages = [
        {"role": "developer", "content": SYSTEM_MSG},
        {"role": "user", "content": game_state},
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages, 
        tools=TOOLS, 
        add_generation_prompt=True, 
        return_dict=True, 
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        out = model.generate(
            **inputs, 
            pad_token_id=tokenizer.eos_token_id, 
            max_new_tokens=64
        )
    
    output = tokenizer.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=False)
    return parse_function_call(output)

# ============================================
# HELPER: Tạo game state từ detection
# ============================================
def create_game_state(suns, has_zombie, can_plant):
    """
    Tạo game state string từ detection results
    
    Args:
        suns: list of (x, y) sun positions, hoặc None/[]
        has_zombie: bool
        can_plant: bool
    """
    if suns and len(suns) > 0:
        x, y = suns[0]  # Lấy sun đầu tiên
        sun_str = f"HAS_SUN x={x} y={y}"
    else:
        sun_str = "NO_SUN"
    
    zombie_str = "HAS_ZOMBIE" if has_zombie else "NO_ZOMBIE"
    plant_str = "CAN_PLANT" if can_plant else "CANNOT_PLANT"
    
    return f"{sun_str}. {zombie_str}. {plant_str}"

# ============================================
# TEST
# ============================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("TEST PVZ BOT")
    print("="*50)
    
    # Test trực tiếp
    tests = [
        "HAS_SUN x=300 y=150. NO_ZOMBIE. CAN_PLANT",
        "NO_SUN. HAS_ZOMBIE. CAN_PLANT",
        "NO_SUN. NO_ZOMBIE. CANNOT_PLANT",
    ]
    
    for t in tests:
        action, args = get_action(t)
        print(f"\nInput: {t}")
        print(f"Action: {action}, Args: {args}")
    
    # Test với helper function
    print("\n" + "="*50)
    print("TEST VỚI HELPER")
    print("="*50)
    
    # Có sun
    gs = create_game_state([(350, 120)], has_zombie=False, can_plant=True)
    action, args = get_action(gs)
    print(f"\nGame state: {gs}")
    print(f"Action: {action}, Args: {args}")
    
    # Không sun, có zombie
    gs = create_game_state([], has_zombie=True, can_plant=True)
    action, args = get_action(gs)
    print(f"\nGame state: {gs}")
    print(f"Action: {action}, Args: {args}")
