import os
import json
import shutil
from pathlib import Path

# === 使用者設定區 ===
IMG_ROOT = Path("data/LISA Lights Dataset/vehicle_light_images/")  # 圖片來源目錄
ANNOTATION_FILES = [
    "data/LISA Lights Dataset/vehicle_light_state_annotations/keypoints_vcrop_left_front_train_first500.json",
    "data/LISA Lights Dataset/vehicle_light_state_annotations/keypoints_vcrop_left_front_val_updated.json",
    "data/LISA Lights Dataset/vehicle_light_state_annotations/keypoints_vcrop_left_rear_train_first500.json",
    "data/LISA Lights Dataset/vehicle_light_state_annotations/keypoints_vcrop_left_rear_val_updated.json",
    "data/LISA Lights Dataset/vehicle_light_state_annotations/keypoints_vcrop_right_front_train_first500.json",
    "data/LISA Lights Dataset/vehicle_light_state_annotations/keypoints_vcrop_right_front_val_updated.json",
    "data/LISA Lights Dataset/vehicle_light_state_annotations/keypoints_vcrop_right_rear_train_first500.json",
    "data/LISA Lights Dataset/vehicle_light_state_annotations/keypoints_vcrop_right_rear_val_updated.json"
]

OUTPUT_ROOT = Path("data/lisa_dataset/classification")
SPLIT_MAP = {
    "train": OUTPUT_ROOT / "train",
    "val": OUTPUT_ROOT / "val",
    "test": OUTPUT_ROOT / "val"  # 測試集一併當作驗證用
}

SIGNAL_MAP = {
    "left_turn_signal": "left_signal",
    "right_turn_signal": "right_signal",
    "off": "no_signal",
    "braking": "brake_signal",
    "unknown": "unknown_signal"
}

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# === 主程式 ===
for anno_path in ANNOTATION_FILES:
    split = "train" if "train" in anno_path else "val" if "val" in anno_path else "test"
    print(f"處理標註檔: {anno_path} (=> {split})")

    # 根據標註檔名推測對應的圖片子目錄
    # 例如 keypoints_vcrop_left_front_train.json -> train_left_front
    basename = Path(anno_path).stem.replace("keypoints_vcrop_", "")
#    image_subdir = basename.replace("_val", "val").replace("_test", "test").replace("_train", "train")
    image_subdir = basename.replace("_updated","").replace("_first500", "")
    img_dir = IMG_ROOT / image_subdir

    with open(anno_path, "r") as f:
        entries = json.load(f)

    for entry in entries:
        image_id = entry["image_id"] + ".jpg"
        signal = entry.get("signal", "unknown").lower()
        label = SIGNAL_MAP.get(signal, "no_signal")

        src_path = img_dir / image_id
        dst_dir = SPLIT_MAP[split] / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / image_id

        if not src_path.exists():
            print(f"⚠️ 找不到圖片: {src_path}")
            continue

        shutil.copy(src_path, dst_path)

print("✅ 圖片分類與搬移完成！")
