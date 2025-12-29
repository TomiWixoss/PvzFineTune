# -*- coding: utf-8 -*-
"""
Gemini Key Manager - Quản lý và xoay vòng API keys
Tự động chuyển key khi gặp rate limit (429) hoặc overload (503)
"""

import os
from typing import Optional
from dotenv import load_dotenv
from google import genai

load_dotenv()


class GeminiKeyManager:
    """Quản lý nhiều API keys, tự động rotate khi lỗi"""
    
    def __init__(self, keys: Optional[list[str]] = None):
        self.keys = keys or self._load_keys_from_env()
        if not self.keys:
            raise ValueError("No API keys found. Set GEMINI_API_KEY in .env")
        
        self.current_index = 0
        self.blocked_keys: set[int] = set()  # Keys bị block (rate limit)
        self._clients: dict[int, genai.Client] = {}
        
        print(f"🔑 Loaded {len(self.keys)} API key(s)")
    
    def _load_keys_from_env(self) -> list[str]:
        """Load keys từ env, hỗ trợ comma-separated hoặc GEMINI_API_KEY_1, _2, ..."""
        keys = []
        
        # Cách 1: Comma-separated
        main_key = os.environ.get("GEMINI_API_KEY", "")
        if main_key:
            keys.extend([k.strip() for k in main_key.split(",") if k.strip()])
        
        # Cách 2: GEMINI_API_KEY_1, GEMINI_API_KEY_2, ...
        i = 1
        while True:
            key = os.environ.get(f"GEMINI_API_KEY_{i}")
            if not key:
                break
            keys.append(key.strip())
            i += 1
        
        return list(set(keys))  # Remove duplicates
    
    def get_client(self) -> genai.Client:
        """Lấy client với key hiện tại"""
        if self.current_index not in self._clients:
            self._clients[self.current_index] = genai.Client(
                api_key=self.keys[self.current_index]
            )
        return self._clients[self.current_index]
    
    def get_current_key_info(self) -> str:
        """Lấy thông tin key hiện tại (masked)"""
        key = self.keys[self.current_index]
        return f"#{self.current_index + 1}/{len(self.keys)} ({key[:8]}...)"
    
    def rotate_key(self) -> bool:
        """
        Chuyển sang key tiếp theo
        Returns: True nếu còn key khả dụng, False nếu hết
        """
        self.blocked_keys.add(self.current_index)
        
        # Tìm key chưa bị block
        for i in range(len(self.keys)):
            next_index = (self.current_index + 1 + i) % len(self.keys)
            if next_index not in self.blocked_keys:
                self.current_index = next_index
                print(f"🔄 Rotated to key {self.get_current_key_info()}")
                return True
        
        print("❌ All keys are blocked!")
        return False
    
    def reset_blocked(self):
        """Reset danh sách keys bị block"""
        self.blocked_keys.clear()
    
    def has_available_key(self) -> bool:
        """Kiểm tra còn key khả dụng không"""
        return len(self.blocked_keys) < len(self.keys)


def is_rate_limit_error(error: Exception) -> bool:
    """Check if error is 429 rate limit"""
    error_str = str(error).lower()
    return "429" in error_str or "rate" in error_str or "quota" in error_str


def is_overload_error(error: Exception) -> bool:
    """Check if error is 503 overload"""
    error_str = str(error).lower()
    return "503" in error_str or "overload" in error_str or "unavailable" in error_str


def is_retryable_error(error: Exception) -> bool:
    """Check if error should trigger retry"""
    return is_rate_limit_error(error) or is_overload_error(error)
