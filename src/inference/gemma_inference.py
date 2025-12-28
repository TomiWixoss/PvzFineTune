# -*- coding: utf-8 -*-
"""
FunctionGemma inference for PvZ bot decisions
"""

import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GEMMA_MODEL_PATH

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.utils import get_json_schema
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class GemmaInference:
    """FunctionGemma model for game action decisions"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or GEMMA_MODEL_PATH
        self.model = None
        self.tokenizer = None
        self.device = None
        self.tools = None
        self.system_msg = "You are a PvZ game bot. Choose ONE action based on game state."
    
    def load(self) -> bool:
        """Load FunctionGemma model"""
        if not HAS_TRANSFORMERS:
            print("✗ transformers not installed. Run: pip install torch transformers")
            return False
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            print(f"Loading FunctionGemma: {self.model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                torch_dtype=torch.float32,
                local_files_only=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True
            )
            self.model = self.model.to(self.device)
            self._setup_tools()
            print(f"✓ Model loaded on {self.device}")
            return True
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False
    
    def _setup_tools(self):
        """Define available tools/actions"""
        def collect_sun(x: int, y: int) -> str:
            """
            Click to collect sun at pixel position.
            
            Args:
                x: X coordinate of sun position in pixels
                y: Y coordinate of sun position in pixels
            """
            return "Collected"

        def plant_pea_shooter(row: int, col: int) -> str:
            """
            Plant a pea shooter at grid position.
            
            Args:
                row: Row index 0-4 (0=top, 4=bottom)
                col: Column index 0-8 (0=left, 8=right)
            """
            return "Planted"

        def do_nothing() -> str:
            """Wait and do nothing."""
            return "Waiting"
        
        self.tools = [
            get_json_schema(collect_sun), 
            get_json_schema(plant_pea_shooter), 
            get_json_schema(do_nothing)
        ]
    
    def _parse_function_call(self, output: str):
        """Parse function call from model output"""
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
    
    def get_action(self, game_state: str):
        """Get action from game state"""
        messages = [
            {"role": "developer", "content": self.system_msg},
            {"role": "user", "content": game_state},
        ]
        
        inputs = self.tokenizer.apply_chat_template(
            messages, 
            tools=self.tools, 
            add_generation_prompt=True, 
            return_dict=True, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            out = self.model.generate(
                **inputs, 
                pad_token_id=self.tokenizer.eos_token_id, 
                max_new_tokens=32,  # Giảm từ 64 xuống 32 (đủ cho function call)
                do_sample=False,    # Greedy decoding - nhanh hơn
                use_cache=True      # Enable KV cache
            )
        
        output = self.tokenizer.decode(
            out[0][len(inputs["input_ids"][0]):], 
            skip_special_tokens=False
        )
        return self._parse_function_call(output)
    
    def should_plant(self, has_zombie: bool, can_plant: bool) -> bool:
        """Decide if should plant"""
        if not can_plant:
            return False
        
        game_state = f"NO_SUN. {'HAS_ZOMBIE' if has_zombie else 'NO_ZOMBIE'}. CAN_PLANT"
        action, _ = self.get_action(game_state)
        return action == "plant_pea_shooter"
    
    @staticmethod
    def create_game_state(suns: list, zombies: list, plants: list, can_plant: bool) -> str:
        """Create game state string from detection results"""
        # Sun info
        if suns and len(suns) > 0:
            x, y = suns[0] if isinstance(suns[0], tuple) else (suns[0]["x"], suns[0]["y"])
            sun_str = f"HAS_SUN x={x} y={y}"
        else:
            sun_str = "NO_SUN"
        
        # Zombie info with row
        if zombies and len(zombies) > 0:
            # Find which row zombie is in
            zombie = zombies[0] if isinstance(zombies[0], dict) else {"x": zombies[0][0], "y": zombies[0][1]}
            zombie_row = GemmaInference._get_row_from_y(zombie["y"])
            zombie_str = f"HAS_ZOMBIE row={zombie_row}"
        else:
            zombie_str = "NO_ZOMBIE"
        
        # Plant count per row
        plant_str = "CAN_PLANT" if can_plant else "CANNOT_PLANT"
        
        return f"{sun_str}. {zombie_str}. {plant_str}"
    
    @staticmethod
    def _get_row_from_y(y: int) -> int:
        """Get row index from y coordinate"""
        from config import GRID_ROWS_Y
        min_dist = float('inf')
        closest_row = 0
        for i, row_y in enumerate(GRID_ROWS_Y):
            dist = abs(y - row_y)
            if dist < min_dist:
                min_dist = dist
                closest_row = i
        return closest_row


def main():
    """Test inference"""
    print("\n" + "="*50)
    print("TEST PVZ BOT")
    print("="*50)
    
    bot = GemmaInference()
    if not bot.load():
        return
    
    tests = [
        "HAS_SUN x=300 y=150. NO_ZOMBIE. CAN_PLANT",
        "NO_SUN. HAS_ZOMBIE. CAN_PLANT",
        "NO_SUN. NO_ZOMBIE. CANNOT_PLANT",
    ]
    
    for t in tests:
        action, args = bot.get_action(t)
        print(f"\nInput: {t}")
        print(f"Action: {action}, Args: {args}")


if __name__ == "__main__":
    main()
