# Prompt: AI xem video Plants vs Zombies tạo danh sách Action + Timestamp

## Gửi cho AI (GPT-4V, Claude, Gemini) kèm video gameplay:

---

Bạn là chuyên gia phân tích gameplay Plants vs Zombies. Xem video này và ghi lại TẤT CẢ hành động người chơi thực hiện theo thời gian.

## HÀNH ĐỘNG CẦN GHI NHẬN:

### 1. `collect_sun` - Thu hoạch mặt trời

- Khi người chơi click vào sun (mặt trời vàng rơi từ trời hoặc từ sunflower)
- Ghi lại vị trí pixel gần đúng nếu có thể

### 2. `plant_pea_shooter` - Trồng cây bắn đậu

- Khi người chơi click seed packet rồi đặt cây xuống ô
- Ghi lại row (0-4, từ trên xuống) và col (0-8, từ trái qua)

### 3. `do_nothing` - Không làm gì

- Khoảng thời gian người chơi chờ đợi (không click gì)
- Ghi lại mỗi 2-3 giây nếu không có action

## GRID LAYOUT:

```
        Col: 0    1    2    3    4    5    6    7    8
Row 0:  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]
Row 1:  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]
Row 2:  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]
Row 3:  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]
Row 4:  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]
```

## OUTPUT FORMAT (JSON):

```json
[
  { "time": "0:03", "action": "collect_sun", "args": { "x": 400, "y": 200 } },
  {
    "time": "0:05",
    "action": "plant_pea_shooter",
    "args": { "row": 2, "col": 0 }
  },
  { "time": "0:08", "action": "collect_sun", "args": {} },
  { "time": "0:10", "action": "do_nothing", "args": {} },
  { "time": "0:15", "action": "collect_sun", "args": {} },
  {
    "time": "0:18",
    "action": "plant_pea_shooter",
    "args": { "row": 1, "col": 0 }
  },
  {
    "time": "0:25",
    "action": "plant_pea_shooter",
    "args": { "row": 3, "col": 1 }
  }
]
```

## QUY TẮC:

1. Ghi CHÍNH XÁC thời điểm action xảy ra (MM:SS hoặc M:SS)
2. Mỗi lần click sun = 1 action `collect_sun`
3. Mỗi lần trồng cây = 1 action `plant_pea_shooter` với row/col
4. Nếu không chắc vị trí pixel của sun, để args rỗng `{}`
5. Ưu tiên ghi `collect_sun` và `plant_pea_shooter`, chỉ ghi `do_nothing` khi cần thiết
6. Sắp xếp theo thứ tự thời gian tăng dần

## VÍ DỤ PHÂN TÍCH:

- 0:03 - Thấy sun rơi, người chơi click → `collect_sun`
- 0:05 - Click seed packet, đặt cây hàng giữa cột đầu → `plant_pea_shooter` row=2, col=0
- 0:08 - Sun từ sunflower, click thu → `collect_sun`
- 0:12 - Zombie xuất hiện, người chơi chờ → `do_nothing`
- 0:15 - Đủ sun, trồng cây hàng trên → `plant_pea_shooter` row=1, col=1

---

Bây giờ xem video và tạo danh sách actions JSON:
