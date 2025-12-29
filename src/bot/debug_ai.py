# -*- coding: utf-8 -*-
"""
Debug wrapper - Chạy auto_play với logging game state
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.auto_play import PvZAutoPlay
from inference.gemma_inference import GemmaInference


class DebugAutoPlay(PvZAutoPlay):
    """Wrap PvZAutoPlay với logging"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logs = []
    
    def run(self):
        """Override run để thêm logging"""
        # Monkey patch get_action để log
        original_get_action = self.ai.get_action
        
        def logged_get_action(game_state):
            action, args = original_get_action(game_state)
            
            log_entry = {
                "time": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                "game_state": game_state,
                "action": action,
                "args": args
            }
            self.logs.append(log_entry)
            
            print(f"\n{'='*60}")
            print(f"[{log_entry['time']}] GAME STATE: {game_state}")
            print(f">>> AI DECISION: {action}({args})")
            print(f"{'='*60}")
            
            return action, args
        
        self.ai.get_action = logged_get_action
        
        try:
            super().run()
        finally:
            # Save logs
            log_file = f"data/ai_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(self.logs, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Logs saved: {log_file} ({len(self.logs)} entries)")


if __name__ == "__main__":
    DebugAutoPlay().run()
