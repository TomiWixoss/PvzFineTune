# PvZ AI Bot

AI bot tự động chơi Plants vs Zombies sử dụng YOLOv11 (object detection) và FunctionGemma (decision making).

## Cấu trúc thư mục

```
pvz-ai-bot/
├── pvz_ai/                    # Main package
│   ├── core/                  # Constants và config
│   │   ├── constants.py       # Hằng số không đổi
│   │   └── config.py          # Cấu hình có thể thay đổi
│   ├── utils/                 # Utilities
│   │   ├── time_utils.py      # Parse/format timestamps
│   │   ├── grid_utils.py      # Grid coordinate conversion
│   │   └── window_capture.py  # Capture game window
│   ├── inference/             # AI inference
│   │   ├── yolo_detector.py   # YOLO detection (OpenVINO)
│   │   └── gemma_inference.py # Gemma inference (OpenVINO)
│   ├── bot/                   # Auto play bot
│   │   ├── controller.py      # Game controller
│   │   └── auto_play.py       # Main bot
│   ├── data/                  # Data collection
│   │   ├── youtube_downloader.py
│   │   ├── video_to_frames.py
│   │   ├── video_dataset_builder.py
│   │   └── dataset_converter.py
│   ├── labeler/               # AI video labeling
│   │   ├── gemini_client.py   # Gemini API key manager
│   │   ├── validator.py       # Action validator
│   │   ├── auto_fixer.py      # Auto fix timestamps
│   │   └── ai_labeler.py      # Main AI labeler
│   └── tools/                 # Utility tools
│       ├── realtime_detection.py
│       └── get_positions.py
├── data/
│   ├── raw/                   # Raw data (videos, frames)
│   ├── processed/             # Processed datasets
│   └── ai_labeler/            # AI labeler outputs
├── models/
│   ├── yolo/                  # YOLO model (best.xml)
│   └── gemma/                 # Gemma model
├── notebooks/
│   ├── yolo_finetune.ipynb
│   └── gemma_finetune.ipynb
└── requirements.txt
```

## Cài đặt

```bash
pip install -r requirements.txt

# Cài ffmpeg (cần cho cắt video YouTube)
winget install ffmpeg
```

## Commands

### Data Collection

```bash
# Tải video YouTube
python -m pvz_ai.data.youtube_downloader "https://youtu.be/VIDEO_ID" -s 1:00 -e 5:00 -n pvz_level1

# Tách frames
python -m pvz_ai.data.video_to_frames data/raw/videos/pvz_level1.mp4 -f 1

# AI tự động label video
python -m pvz_ai.labeler.ai_labeler video.mp4

# Build dataset từ actions file
python -m pvz_ai.data.video_dataset_builder video.mp4 -a actions.json -o dataset.json

# Convert sang training format
python -m pvz_ai.data.dataset_converter dataset.json -o training_data.json
```

### Inference & Bot

```bash
# Detection realtime từ game
python -m pvz_ai.tools.realtime_detection

# Chạy bot tự động
python -m pvz_ai.bot.auto_play

# Lấy tọa độ grid
python -m pvz_ai.tools.get_positions
```

## Classes được detect (YOLO)

| Class                  | Mô tả                           |
| ---------------------- | ------------------------------- |
| `sun`                  | Mặt trời                        |
| `zombie`               | Zombie                          |
| `pea_shooter`          | Cây bắn đậu đã trồng            |
| `pea_shooter_ready`    | Seed packet sáng (có thể trồng) |
| `pea_shooter_cooldown` | Seed packet tối (đang cooldown) |
| `sunflower_reward`     | Phần thưởng qua màn             |

## AI Actions

- `plant(plant_type, row, col)` - Trồng cây tại vị trí grid
- `wait()` - Chờ (khi seed cooldown hoặc không đủ sun)

Thu thập sun được xử lý bằng rule-based.

## Training

Mở notebooks trên Google Colab (chọn GPU runtime):

- `notebooks/yolo_finetune.ipynb` - Train YOLO
- `notebooks/gemma_finetune.ipynb` - Train Gemma

## Environment Variables

```bash
# .env
GEMINI_API_KEY=your_api_key_here
# Hoặc nhiều keys
GEMINI_API_KEY=key1,key2,key3
```
