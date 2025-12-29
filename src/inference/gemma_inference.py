# -*- coding: utf-8 -*-
"""
FunctionGemma inference for PvZ bot decisions - OpenVINO version
"""

import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GEMMA_MODEL_PATH

try:
    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer
    from transformers.utils import get_json_schema
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False


class GemmaInference:
    """FunctionGemma model for game action decisions - OpenVINO"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or GEMMA_MODEL_PATH
        self.model = None
        self.tokenizer = None
        self.tools = None
        self.system_msg = """PvZ bot. Choose action based on game state.
- PLANTS: planted plants (type,row,col)
- ZOMBIES: zombies (type,row,col)
- SEEDS: seed packets (type,status)
Plant when seed ready. Wait when cooldown."""
    
    def load(self) -> bool:
        """Load OpenVINO model"""
        if not HAS_OPENVINO:
            print("✗ optimum-intel not installed. Run: pip install optimum[openvino]")
            return False
        
        try:
            print(f"Loading Gemma OpenVINO: {self.model_path}")
            self.model = OVModelForCausalLM.from_pretrained(
                self.model_path,
                local_files_only=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True
            )
            self._setup_tools()
            print(f"✓ OpenVINO model loaded")
            return True
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False
    
    def _setup_tools(self):
        """Define available tools/actions"""
        def plant(plant_type: str, row: int, col: int) -> str:
            """
            Plant a plant at grid position.
            
            Args:
                plant_type: Type of plant to place (pea_shooter, sunflower, wall_nut)
                row: Row index from 0 to 4 (0=top, 4=bottom)
                col: Column index from 0 to 8 (0=left, 8=right)
            """
            return "Planted"

        def wait() -> str:
            """Wait and do nothing. Use when seed is on cooldown or not enough sun."""
            return "Waiting"
        
        self.tools = [get_json_schema(plant), get_json_schema(wait)]
    
    def _parse_function_call(self, output: str):
        """Parse function call from model output"""
        # Format: call:plant{col:2,plant_type:<escape>pea_shooter<escape>,row:2}
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
                    k = k.strip()
                    v = v.strip()
                    # Remove <escape> tags
                    v = v.replace("<escape>", "")
                    try:
                        args[k] = int(v)
                    except ValueError:
                        args[k] = v
        
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
        )
        
        out = self.model.generate(
            **inputs, 
            pad_token_id=self.tokenizer.eos_token_id, 
            max_new_tokens=64
        )
        
        output = self.tokenizer.decode(
            out[0][len(inputs["input_ids"][0]):], 
            skip_special_tokens=False
        )
        return self._parse_function_call(output)
    
    @staticmethod
    def create_game_state(plants: list, zombies: list, seeds: list) -> str:
        """
        Create game state string from detection results
        
        Args:
            plants: List of {"type": "pea_shooter", "row": 2, "col": 0, ...}
            zombies: List of {"type": "zombie", "row": 2, "col": 8, ...}
            seeds: List of {"type": "pea_shooter", "status": "ready/cooldown", ...}
        
        Returns:
            "PLANTS:[(pea_shooter,2,0)]. ZOMBIES:[(zombie,2,8)]. SEEDS:[(pea_shooter,ready)]"
        """
        parts = []
        
        # Plants
        if plants:
            plant_str = ",".join([f"({p.get('type','plant')},{p.get('row',0)},{p.get('col',0)})" for p in plants])
            parts.append(f"PLANTS:[{plant_str}]")
        else:
            parts.append("PLANTS:[]")
        
        # Zombies
        if zombies:
            zombie_str = ",".join([f"({z.get('type','zombie')},{z.get('row',0)},{z.get('col',8)})" for z in zombies])
            parts.append(f"ZOMBIES:[{zombie_str}]")
        else:
            parts.append("ZOMBIES:[]")
        
        # Seeds
        if seeds:
            seed_str = ",".join([f"({s.get('type','unknown')},{s.get('status','unknown')})" for s in seeds])
            parts.append(f"SEEDS:[{seed_str}]")
        else:
            parts.append("SEEDS:[]")
        
        return ". ".join(parts)


def main():
    """Test inference"""
    print("\n" + "="*50)
    print("TEST PVZ BOT (OpenVINO)")
    print("="*50)
    
    bot = GemmaInference()
    if not bot.load():
        return
    
    tests = [
        "PLANTS:[]. ZOMBIES:[]. SEEDS:[(pea_shooter,ready)]",
        "PLANTS:[(pea_shooter,2,0)]. ZOMBIES:[(zombie,2,7)]. SEEDS:[(pea_shooter,cooldown)]",
        "PLANTS:[(pea_shooter,2,0),(pea_shooter,2,1)]. ZOMBIES:[(zombie,1,6)]. SEEDS:[(pea_shooter,ready)]",
    ]
    
    for t in tests:
        action, args = bot.get_action(t)
        print(f"\nInput: {t}")
        print(f"Action: {action}, Args: {args}")


if __name__ == "__main__":
    main()
