Vehicle_Direction_Light/
├── data/                      # ✅ 已標記序列儲存區
│   ├── LISA Lights Dataset/     # 原始資料集
│   └── lisa_dataset/
│       └── classification
│           ├── train/
│           │   ├── left_signal/
│           │   ├── no_signal/
│           │   └── right_signal/
│           │
│           └── val/
│               ├── left_signal/
│               ├── no_signal/
│               └── right_signal/
├── models/
│   └── turn_signal_net.py     # 你的模型架構（例如 MobileNetV2 + LSTM）
│
│── generate_classification_dataset.py # 根據 json 中的標註來分類原始資料並搬移至 lisa_dataset/ 資料夾中
├── train.py                   # 主訓練流程（讀取 data/ 開始訓練）
├── test.py                    # 可選：測試流程
└── README.md