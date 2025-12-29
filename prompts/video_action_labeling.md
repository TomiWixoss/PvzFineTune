# Prompt: AI xem video Plants vs Zombies tạo danh sách Action + Timestamp

## Gửi cho AI (GPT-4V, Claude, Gemini) kèm video gameplay:

---

Bạn là chuyên gia phân tích gameplay Plants vs Zombies. Xem video frame-by-frame và ghi lại hành động TRỒNG CÂY của người chơi.

## ⚠️ LƯU Ý QUAN TRỌNG

- **KHÔNG ghi action thu thập sun** - việc này do code rule tự động xử lý
- **CHỈ ghi 2 loại action**: `plant` (trồng cây) và `wait` (chờ)
- AI sẽ học cách quyết định KHI NÀO và Ở ĐÂU nên trồng cây

## 🎯 2 LOẠI ACTION:

### 1. `plant` - Trồng cây

**KHI NÀO**: Người chơi click seed packet VÀ đặt cây xuống grid
**THAM SỐ**:

- `plant_type`: Loại cây (pea_shooter, sunflower, wall_nut, ...)
- `row`: Hàng (0-4, 0=trên cùng)
- `col`: Cột (0-8, 0=trái nhất)

```json
{
  "time": "0:18",
  "action": "plant",
  "args": { "plant_type": "pea_shooter", "row": 2, "col": 0 },
  "note": "trồng pea_shooter hàng giữa, cột đầu"
}
```

**GRID**:

```
Row 0 (top)    : [ ][ ][ ][ ][ ][ ][ ][ ][ ]
Row 1          : [ ][ ][ ][ ][ ][ ][ ][ ][ ]
Row 2 (middle) : [ ][ ][ ][ ][ ][ ][ ][ ][ ]
Row 3          : [ ][ ][ ][ ][ ][ ][ ][ ][ ]
Row 4 (bottom) : [ ][ ][ ][ ][ ][ ][ ][ ][ ]
                 Col 0 → → → → → → → → Col 8
```

**PLANT TYPES** (phổ biến):

- `pea_shooter` - Bắn đậu
- `sunflower` - Hoa hướng dương
- `wall_nut` - Hạt óc chó (chắn)
- `cherry_bomb` - Bom cherry
- `snow_pea` - Đậu băng
- `repeater` - Bắn đậu đôi

### 2. `wait` - Chờ

**KHI NÀO**:

- Seed packet đang cooldown (xám)
- Không đủ sun để trồng
- Đang chờ zombie xuất hiện
- Không cần trồng thêm

```json
{
  "time": "0:25",
  "action": "wait",
  "args": {},
  "note": "seed cooldown, chờ"
}
```

## 📋 QUY TRÌNH XEM VIDEO:

```
1. Play video với tốc độ 0.5x hoặc 0.25x

2. Mỗi khi thấy NGƯỜI CHƠI TRỒNG CÂY:
   - Pause ngay
   - Ghi timestamp
   - Xác định loại cây (plant_type)
   - Xác định vị trí (row, col)
   - Action: plant

3. Mỗi 3-5 giây không trồng gì:
   - Ghi wait
   - Note lý do (cooldown, chờ sun, ...)
```

## ✅ VALIDATION CHECKLIST:

| Action  | Điều kiện BẮT BUỘC                                  |
| ------- | --------------------------------------------------- |
| `plant` | Người chơi THỰC SỰ trồng cây tại timestamp đó       |
| `wait`  | Không có hành động trồng cây trong khoảng thời gian |

## ❌ LỖI THƯỜNG GẶP:

```json
// ❌ SAI: Ghi collect_sun (không dùng nữa!)
{"time": "0:15", "action": "collect_sun"}

// ❌ SAI: Thiếu plant_type
{"time": "0:20", "action": "plant", "args": {"row": 2, "col": 1}}

// ❌ SAI: Ghi plant khi chưa thực sự trồng
{"time": "0:20", "action": "plant", "args": {"plant_type": "pea_shooter", "row": 2, "col": 1}}
// Người chơi chỉ click seed nhưng chưa đặt xuống
```

## ✅ VÍ DỤ ĐÚNG:

```json
[
  {
    "time": "0:05",
    "action": "wait",
    "args": {},
    "note": "game starting, chờ đủ sun"
  },
  {
    "time": "0:09",
    "action": "plant",
    "args": { "plant_type": "pea_shooter", "row": 2, "col": 0 },
    "note": "trồng pea_shooter đầu tiên"
  },
  {
    "time": "0:15",
    "action": "wait",
    "args": {},
    "note": "seed cooldown"
  },
  {
    "time": "0:22",
    "action": "plant",
    "args": { "plant_type": "pea_shooter", "row": 2, "col": 1 },
    "note": "trồng thêm pea_shooter"
  },
  {
    "time": "0:30",
    "action": "wait",
    "args": {},
    "note": "chờ sun"
  },
  {
    "time": "0:41",
    "action": "plant",
    "args": { "plant_type": "pea_shooter", "row": 1, "col": 0 },
    "note": "zombie xuất hiện row 1, trồng phòng thủ"
  },
  {
    "time": "0:50",
    "action": "wait",
    "args": {},
    "note": "đang phòng thủ tốt"
  },
  {
    "time": "0:58",
    "action": "plant",
    "args": { "plant_type": "wall_nut", "row": 2, "col": 3 },
    "note": "đặt wall_nut chắn zombie"
  }
]
```

## 🎬 OUTPUT FORMAT:

```json
[
  {
    "time": "M:SS",
    "action": "plant | wait",
    "args": { "plant_type": "...", "row": N, "col": N },
    "note": "lý do action"
  }
]
```

**Time format**: `M:SS` hoặc `M:SS.S` (ví dụ: `0:12` hoặc `0:12.5`)

---

## BÂY GIỜ XEM VIDEO VÀ TẠO DANH SÁCH:

⚠️ Nhớ:

1. **CHỈ ghi `plant` và `wait`** - KHÔNG ghi collect_sun
2. **`plant` phải có đủ**: plant_type, row, col
3. **Ghi timestamp chính xác** khi người chơi đặt cây xuống
4. **Note** lý do để hiểu context (zombie ở đâu, tại sao trồng vị trí đó)
