import torch
from torchvision import models, transforms
from PIL import Image
import os
import argparse

# === 分類對應 ===
class_names = ['no_signal', 'left_signal', 'right_signal']

# === 模型載入 ===
def load_model(model_path):
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.last_channel, 3)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# === 圖片預處理 ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === 推論單張圖片 ===
def predict_image(model, image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 增加 batch 維度
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        label = class_names[predicted.item()]
    return label

# === 主程式 ===
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='單張圖片推論路徑')
    parser.add_argument('--dir', type=str, help='整個資料夾推論')
    parser.add_argument('--model', type=str, default='turn_signal_model.pth', help='模型檔路徑')
    args = parser.parse_args()

    model = load_model(args.model)

    if args.image:
        if not os.path.exists(args.image):
            print("❌ 找不到圖片路徑 (--image)")
            exit(1)
        label = predict_image(model, args.image)
        print(f"✅ 圖片：{args.image} → 推論結果：{label}")

    elif args.dir:
        if not os.path.exists(args.dir):
            print("❌ 找不到資料夾路徑 (--dir)")
            exit(1)

        print(f"📂 批次推論資料夾：{args.dir}\n")
        for fname in os.listdir(args.dir):
            fpath = os.path.join(args.dir, fname)
            if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            try:
                label = predict_image(model, fpath)
                print(f"  {fname:30s} → {label}")
            except Exception as e:
                print(f"  ⚠️ 無法處理 {fname}: {e}")

    else:
        print("❗請使用 --image 或 --dir 參數提供圖片或資料夾路徑")