# -*- coding: utf-8 -*-
"""
System prompts cho Gemini AI Labeler
Đọc labels từ YOLO metadata để không hardcode
"""

import yaml
from pathlib import Path


def load_yolo_labels(metadata_path: str = "models/yolo/metadata.yaml") -> dict:
    """Load labels từ YOLO metadata.yaml"""
    path = Path(metadata_path)
    if not path.exists():
        return {}
    
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    return data.get('names', {})


def get_plant_types(labels: dict) -> list:
    """Lọc ra các loại plant từ labels (bỏ _cooldown, _ready, sun, zombie, ...)"""
    plant_types = []
    exclude_suffixes = ['_cooldown', '_ready', '_reward']
    exclude_names = ['sun', 'zombie', 'zombies']
    
    for idx, name in labels.items():
        # Bỏ qua các label không phải plant
        if name in exclude_names:
            continue
        if any(name.endswith(suffix) for suffix in exclude_suffixes):
            continue
        plant_types.append(name)
    
    return plant_types


def build_system_prompt(metadata_path: str = "models/yolo/metadata.yaml") -> str:
    """Build system prompt với labels từ YOLO metadata"""
    labels = load_yolo_labels(metadata_path)
    plant_types = get_plant_types(labels)
    
    # Format plant types cho prompt
    if plant_types:
        plant_list = ", ".join(plant_types)
    else:
        plant_list = "pea_shooter, sunflower, wall_nut, cherry_bomb, potato_mine, snow_pea, chomper, repeater"
    
    return f"""---
Bạn là chuyên gia phân tích gameplay Plants vs Zombies. Xem video frame-by-frame và ghi lại hành động TRỒNG CÂY của người chơi.

## ⚠️ LƯU Ý QUAN TRỌNG
- **KHÔNG ghi action thu thập sun** - việc này do code rule tự động xử lý
- **CHỈ ghi 2 loại action**: `plant` (trồng cây) và `wait` (chờ)
- **TIMESTAMP CHÍNH XÁC**: Ghi tới millisecond (M:SS.mmm)

## ⏱️ TIMESTAMP FORMAT (BẮT BUỘC):
Format: `M:SS.mmm` (phút:giây.miligiây)
- M = phút (0, 1, 2, ...)
- SS = giây (00-59)
- mmm = miligiây (000-999)

Ví dụ:
- `0:05.250` = 5 giây 250ms
- `0:18.500` = 18 giây 500ms  
- `1:02.750` = 1 phút 2 giây 750ms
- `2:30.125` = 2 phút 30 giây 125ms

⚠️ PHẢI ghi đủ 3 chữ số miligiây!

## 🌱 CÁC LOẠI CÂY (từ YOLO model):
{plant_list}

## 🎯 2 LOẠI ACTION:

### 1. `plant` - Trồng cây
**THAM SỐ**:
- `plant_type`: Loại cây ({plant_list})
- `row`: Hàng (0-4, 0=trên cùng)
- `col`: Cột (0-8, 0=trái nhất)

**GRID**:
```
Row 0 (top)    : [ ][ ][ ][ ][ ][ ][ ][ ][ ]
Row 1          : [ ][ ][ ][ ][ ][ ][ ][ ][ ]
Row 2 (middle) : [ ][ ][ ][ ][ ][ ][ ][ ][ ]
Row 3          : [ ][ ][ ][ ][ ][ ][ ][ ][ ]
Row 4 (bottom) : [ ][ ][ ][ ][ ][ ][ ][ ][ ]
Col 0 → → → → → → → → Col 8
```

### 2. `wait` - Chờ (seed cooldown, không đủ sun, ...)

## 🎬 OUTPUT FORMAT:
```json
[
  {{"time": "0:18.500", "action": "plant", "args": {{"plant_type": "pea_shooter", "row": 2, "col": 0}}, "note": "..."}},
  {{"time": "0:25.250", "action": "wait", "args": {{}}, "note": "..."}}
]
```

⚠️ CHỈ trả về JSON array, không text khác.
⚠️ Timestamp PHẢI có millisecond (M:SS.mmm)
"""


# Lazy load - chỉ build khi cần
_SYSTEM_PROMPT = None

def get_system_prompt() -> str:
    """Get system prompt (cached)"""
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        _SYSTEM_PROMPT = build_system_prompt()
    return _SYSTEM_PROMPT


# Backward compatibility
SYSTEM_PROMPT = property(lambda self: get_system_prompt())


CORRECTION_PROMPT_TEMPLATE = """
Kết quả validation KHÔNG ĐẠT (score: {score:.1f}%).

## LỖI CẦN SỬA:
{error_feedback}

## TRẠNG THÁI GAME TẠI CÁC TIMESTAMP LỖI:
{game_states_info}

## YÊU CẦU:
1. Xem lại video (bạn đã xem ở lượt trước)
2. Dựa vào game_state ở trên để hiểu:
   - PLANTS: cây đã trồng ở đâu (không được trồng chồng)
   - SEEDS: seed packet nào ready/cooldown
3. **LƯU Ý**: Có thể bạn đã ghi THỪA action (video chỉ trồng 3 cây mà bạn ghi 4). Hãy xem lại và XÓA action không có thật.
4. Sửa các lỗi:
   - Không trồng chồng lên ô đã có cây
   - row trong range 0-4, col trong range 0-8
   - CHỈ plant khi seed packet READY (không cooldown)
   - Timestamp phải chính xác khi cây THỰC SỰ được đặt xuống
5. **TIMESTAMP FORMAT**: M:SS.mmm (phút:giây.miligiây, VD: 0:18.500)
6. Trả về JSON array đã sửa
"""
