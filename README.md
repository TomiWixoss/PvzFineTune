# PvZ AI Bot

AI bot tự động chơi Plants vs Zombies sử dụng YOLOv11 (object detection) và FunctionGemma (decision making).

## Cấu trúc thư mục

```
pvz-ai-bot/
├── src/
│   ├── config.py              # Cấu hình chung (class names, paths, colors)
│   ├── bot/
│   │   └── auto_play.py       # Bot tự động chơi game
│   ├── data/
│   │   ├── youtube_downloader.py   # Tải video YouTube
│   │   └── video_to_frames.py      # Tách frames từ video
│   ├── inference/
│   │   ├── yolo_detector.py        # YOLO detection
│   │   ├── gemma_inference.py      # Gemma AI decisions
│   │   ├── realtime_detection.py   # Detection realtime từ game
│   │   └── video_detection.py      # Detection từ video file
│   └── utils/
│       ├── window_capture.py       # Capture cửa sổ game
│       └── get_positions.py        # Tool lấy tọa độ
├── notebooks/
│   ├── yolo_finetune.ipynb    # Train YOLO trên Colab
│   └── gemma_finetune.ipynb   # Train Gemma trên Colab
├── data/
│   └── raw/
│       ├── videos/            # Video gameplay
│       └── frames/            # Frames đã tách
├── models/
│   ├── yolo/pvz_openvino/     # YOLO model (best.xml, best.bin)
│   └── gemma/pvz_functiongemma_final/  # Gemma model
└── requirements.txt
```

## Cài đặt

```bash
pip install -r requirements.txt

# Cài ffmpeg (cần cho cắt video YouTube)
winget install ffmpeg
```

## Workflow

### 1. Thu thập Dataset

```bash
# Tải video YouTube
python -m src.data.youtube_downloader "https://youtu.be/VIDEO_ID" -s 1:00 -e 5:00 -n pvz_level1

# Tách frames (1 FPS)
python -m src.data.video_to_frames data/raw/videos/pvz_level1.mp4

# Tách với FPS khác
python -m src.data.video_to_frames data/raw/videos/pvz_level1.mp4 -f 2

# Tách tất cả videos
python -m src.data.video_to_frames data/raw/videos --batch
```

### 2. Gán nhãn

Upload frames lên [Roboflow](https://roboflow.com/) và gán nhãn với các class:

| Class                  | Mô tả                       |
| ---------------------- | --------------------------- |
| `sun`                  | Mặt trời                    |
| `zombie`               | Zombie                      |
| `pea_shooter`          | Cây bắn đậu đã trồng        |
| `pea_shooter_ready`    | Gói hạt sáng (dùng được)    |
| `pea_shooter_cooldown` | Gói hạt tối (đang cooldown) |
| `sunflower`            | Cây hướng dương đã trồng    |
| `sunflower_ready`      | Gói sunflower sáng          |
| `sunflower_cooldown`   | Gói sunflower tối           |
| `button_continue`      | Nút tiếp tục                |
| `button_start`         | Nút bắt đầu                 |

### 3. Training

Mở notebooks trên Google Colab (chọn GPU runtime):

- `notebooks/yolo_finetune.ipynb` - Train YOLO
- `notebooks/gemma_finetune.ipynb` - Train Gemma

Sau khi train xong, download và giải nén vào thư mục `models/`.

### 4. Chạy Bot

```bash
# Detection realtime từ game
python -m src.inference.realtime_detection

# Detection từ video file
python -m src.inference.video_detection data/raw/videos/pvz_level1.mp4

# Bot tự động chơi
python -m src.bot.auto_play
```

## Commands

| Command                                         | Mô tả                      |
| ----------------------------------------------- | -------------------------- |
| `python -m src.data.youtube_downloader URL`     | Tải video YouTube          |
| `python -m src.data.video_to_frames VIDEO`      | Tách frames từ video       |
| `python -m src.inference.realtime_detection`    | Detection realtime từ game |
| `python -m src.inference.video_detection VIDEO` | Detection từ video file    |
| `python -m src.bot.auto_play`                   | Chạy bot tự động           |
| `python -m src.utils.get_positions`             | Lấy tọa độ trong game      |

## Options

### youtube_downloader

```
-s, --start    Thời gian bắt đầu (MM:SS)
-e, --end      Thời gian kết thúc (MM:SS)
-n, --name     Tên file output
-o, --output   Thư mục output
```

### video_to_frames

```
-f, --fps      FPS để tách (default: 1)
-o, --output   Thư mục output
--batch        Xử lý tất cả videos trong thư mục
```

### video_detection

```
-m, --model    Đường dẫn model YOLO
-c, --conf     Confidence threshold (default: 0.5)
-o, --output   Đường dẫn video output
--show         Hiển thị video trong lúc xử lý
```

### auto_play

```
-m, --model    Đường dẫn model YOLO
-g, --gemma    Đường dẫn model Gemma
```

## Cấu hình

Chỉnh sửa `src/config.py` để thay đổi:

- Đường dẫn models
- Class names và colors
- Tọa độ grid game
- Các thông số bot
