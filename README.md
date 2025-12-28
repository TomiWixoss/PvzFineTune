# PvZ AI Bot

AI bot cho Plants vs Zombies sử dụng YOLOv11 (object detection) và FunctionGemma (decision making).

## Cấu trúc thư mục

```
pvz-ai-bot/
├── src/
│   ├── bot/              # Auto-play bot
│   │   └── auto_play.py  # Main bot (hybrid: rule-based + AI)
│   ├── data/             # Data collection & labeling
│   │   ├── capture_screenshots.py
│   │   ├── collect_training_data.py
│   │   └── auto_label.py
│   ├── inference/        # Model inference
│   │   ├── yolo_detector.py
│   │   ├── gemma_inference.py
│   │   └── realtime_detection.py
│   ├── training/         # Training scripts (Colab)
│   │   ├── yolo_finetune.py
│   │   └── gemma_finetune.py
│   └── utils/            # Utilities
│       ├── window_capture.py
│       └── get_positions.py
├── data/
│   ├── raw/              # Raw screenshots
│   └── processed/        # Training data JSON
├── models/
│   ├── yolo/             # YOLO OpenVINO model
│   └── gemma/            # FunctionGemma model
├── notebooks/            # Colab notebooks
└── requirements.txt
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Sử dụng

### 1. Thu thập Dataset

```bash
# Chụp screenshots tự động
python -m src.data.capture_screenshots

# Thu thập training data với YOLO detection
python -m src.data.collect_training_data
```

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
