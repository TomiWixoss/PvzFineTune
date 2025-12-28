# Prompt: AI xem video Plants vs Zombies tạo danh sách Action + Timestamp

## Gửi cho AI (GPT-4V, Claude, Gemini) kèm video gameplay:

---

Bạn là chuyên gia phân tích gameplay Plants vs Zombies. Xem video frame-by-frame và ghi lại hành động.

## ⚠️ CRITICAL: TIMESTAMP PHẢI CHÍNH XÁC ĐẾN 0.5 GIÂY

Sun chỉ hiển thị 1-2 giây rồi biến mất. Nếu timestamp sai 1 giây = data sai hoàn toàn.

### CÁCH XÁC ĐỊNH TIMESTAMP ĐÚNG:

**Bước 1**: Pause video NGAY LÚC thấy sun/seed sáng
**Bước 2**: Ghi timestamp HIỆN TẠI (không trừ, không cộng)
**Bước 3**: Đó là timestamp cho action

```
VÍ DỤ:
- Pause lúc 0:12.5, thấy sun đang hiển thị → Ghi: "0:12" action: "collect_sun"
- Pause lúc 0:18.0, thấy seed packet sáng → Ghi: "0:18" action: "plant_pea_shooter"
```

## 🎯 3 LOẠI ACTION:

### 1. `collect_sun`

**KHI NÀO**: Thấy sun (vàng tròn) ĐANG HIỂN THỊ trên màn hình
**TIMESTAMP**: Lúc sun đang hiển thị rõ ràng (KHÔNG phải lúc click)

```json
{
  "time": "0:12",
  "action": "collect_sun",
  "args": {},
  "note": "sun visible center screen"
}
```

### 2. `plant_pea_shooter`

**KHI NÀO**: Seed packet SÁNG (có viền sáng, không xám)
**TIMESTAMP**: Lúc seed đang sáng VÀ có đủ sun (50+)

```json
{
  "time": "0:18",
  "action": "plant_pea_shooter",
  "args": { "row": 2, "col": 0 },
  "note": "seed bright, 100 sun"
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

### 3. `do_nothing`

**KHI NÀO**:

- Không có sun trên màn hình
- Seed packet XÁM (cooldown)
- Không đủ sun để trồng

```json
{
  "time": "0:25",
  "action": "do_nothing",
  "args": {},
  "note": "no sun, seed cooldown"
}
```

## 📋 QUY TRÌNH XEM VIDEO:

```
1. Play video với tốc độ 0.5x hoặc 0.25x
2. Mỗi khi thấy SUN xuất hiện:
   - Pause ngay
   - Ghi timestamp
   - Action: collect_sun

3. Mỗi khi thấy SEED SÁNG LÊN:
   - Pause ngay
   - Ghi timestamp
   - Xem người chơi trồng ở đâu (row, col)
   - Action: plant_pea_shooter

4. Mỗi 3-5 giây không có gì:
   - Ghi do_nothing
```

## ✅ VALIDATION CHECKLIST:

Trước khi submit, kiểm tra TỪNG action:

| Action              | Điều kiện BẮT BUỘC                             |
| ------------------- | ---------------------------------------------- |
| `collect_sun`       | Sun PHẢI đang hiển thị tại timestamp đó        |
| `plant_pea_shooter` | Seed PHẢI sáng + đủ sun (50+) tại timestamp đó |
| `do_nothing`        | KHÔNG có sun + seed xám HOẶC không đủ sun      |

## ❌ LỖI THƯỜNG GẶP:

```json
// ❌ SAI: Ghi timestamp sau khi sun biến mất
{"time": "0:15", "action": "collect_sun"}
// Sun xuất hiện 0:12-0:14, biến mất 0:14 → timestamp 0:15 = KHÔNG CÓ SUN

// ❌ SAI: Ghi plant khi seed xám
{"time": "0:20", "action": "plant_pea_shooter", "args": {"row": 2, "col": 1}}
// Seed cooldown từ 0:18-0:25 → timestamp 0:20 = SEED XÁM

// ❌ SAI: Timestamp làm tròn quá nhiều
{"time": "0:10", "action": "collect_sun"}
// Sun xuất hiện 0:12.3 → ghi 0:10 = SAI 2 giây
```

## ✅ VÍ DỤ ĐÚNG:

```json
[
  {
    "time": "0:05",
    "action": "do_nothing",
    "args": {},
    "note": "game starting, no sun yet"
  },
  {
    "time": "0:08",
    "action": "plant_pea_shooter",
    "args": { "row": 2, "col": 0 },
    "note": "first seed ready, 50 sun"
  },
  {
    "time": "0:12",
    "action": "collect_sun",
    "args": {},
    "note": "sun falling from sky, visible now"
  },
  {
    "time": "0:16",
    "action": "do_nothing",
    "args": {},
    "note": "seed cooldown, waiting"
  },
  {
    "time": "0:19",
    "action": "collect_sun",
    "args": {},
    "note": "another sun visible"
  },
  {
    "time": "0:22",
    "action": "plant_pea_shooter",
    "args": { "row": 2, "col": 1 },
    "note": "seed ready again, 100 sun"
  },
  {
    "time": "0:26",
    "action": "do_nothing",
    "args": {},
    "note": "seed cooldown"
  },
  {
    "time": "0:30",
    "action": "collect_sun",
    "args": {},
    "note": "sun from sunflower"
  },
  {
    "time": "0:34",
    "action": "do_nothing",
    "args": {},
    "note": "waiting for sun"
  },
  {
    "time": "0:38",
    "action": "collect_sun",
    "args": {},
    "note": "falling sun visible"
  },
  {
    "time": "0:41",
    "action": "plant_pea_shooter",
    "args": { "row": 2, "col": 2 },
    "note": "seed ready, planting 3rd"
  },
  {
    "time": "0:45",
    "action": "do_nothing",
    "args": {},
    "note": "defending, seed cooldown"
  },
  {
    "time": "0:50",
    "action": "collect_sun",
    "args": {},
    "note": "sun visible"
  },
  { "time": "0:55", "action": "do_nothing", "args": {}, "note": "level ending" }
]
```

## 🎬 OUTPUT FORMAT:

```json
[
  {
    "time": "M:SS",
    "action": "ACTION_TYPE",
    "args": {},
    "note": "why this action"
  }
]
```

**Time format**: `M:SS` hoặc `M:SS.S` (ví dụ: `0:12` hoặc `0:12.5`)

---

## BÂY GIỜ XEM VIDEO VÀ TẠO DANH SÁCH:

⚠️ Nhớ:

1. **Pause video** khi thấy sun/seed sáng
2. **Ghi timestamp chính xác** tại thời điểm pause
3. **Validate** mỗi action trước khi thêm vào list
4. Sun chỉ hiển thị 1-2 giây - timing rất quan trọng!
