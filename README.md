# PvZ AI Bot

AI bot cho Plants vs Zombies sử dụng YOLOv11 (object detection) và FunctionGemma (decision making).

## Cấu trúc thư mục

```
pvz-ai-bot/
├── src/
│   ├── bot/              # Auto-play bot
│   │   └── auto_play.py
│   ├── data/             # Data collection
│   │   ├── youtube_downloader.py
│   │   ├── video_to_frames.py
│   │   └── auto_label.py
│   ├── inference/        # Model inference
│   │   ├── yolo_detector.py
│   │   ├── gemma_inference.py
│   │   └── realtime_detection.py
│   ├── training/         # Training scripts (Colab)
│   │   ├── yolo_finetune.py
│   │   └── gemma_finetune.py
│   └── utils/
│       └── get_positions.py
├── data/
│   ├── raw/              # Videos & frames
│   └── processed/        # Training data JSON
├── models/
│   ├── yolo/             # YOLO OpenVINO model
│   └── gemma/            # FunctionGemma model
└── requirements.txt
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Sử dụng

### 1. Thu thập Dataset từ YouTube

```bash
# Tải video YouTube (full)
python -m src.data.youtube_downloader "https://youtube.com/watch?v=VIDEO_ID"

# Tải video với thời lượng cụ thể (từ 1:00 đến 5:30)
python -m src.data.youtube_downloader "URL" -s 1:00 -e 5:30 -n pvz_gameplay

# Tách frames từ video (1 FPS)
python -m src.data.video_to_frames data/raw/videos/pvz_gameplay.mp4

# Tách frames với FPS khác
python -m src.data.video_to_frames data/raw/videos/pvz_gameplay.mp4 -f 2

# Tách frames từ tất cả videos trong thư mục
python -m src.data.video_to_frames data/raw/videos --batch
```

Sau khi tách frames, dùng [LabelImg](https://github.com/HumanSignal/labelImg), [Roboflow](https://roboflow.com/), hoặc [CVAT](https://www.cvat.ai/) để gán nhãn.

### 2. Training (Google Colab)

Upload các file trong `src/training/` lên Colab và chạy với GPU.

### 3. Realtime Detection

```bash
python -m src.inference.realtime_detection
```

### 4. Auto Play Bot

```bash
python -m src.bot.auto_play
```

## Classes được nhận diện

| Class                | Mô tả                         | Màu     |
| -------------------- | ----------------------------- | ------- |
| pea_shooter          | Cây bắn đậu                   | Xanh lá |
| pea_shooter_pack_no  | Gói hạt giống (chưa sẵn sàng) | Cam     |
| pea_shooter_pack_yes | Gói hạt giống (sẵn sàng)      | Vàng    |
| sun                  | Mặt trời                      | Cyan    |
| zombie               | Zombie                        | Đỏ      |

## Cấu hình

Đường dẫn model mặc định:

- YOLO: `models/yolo/pvz_openvino/best.xml`
- Gemma: `models/gemma/pvz_functiongemma_final/`

## Utilities

```bash
# Lấy tọa độ trong game
python -m src.utils.get_positions
```
