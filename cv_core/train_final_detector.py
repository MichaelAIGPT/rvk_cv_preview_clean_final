import os
import torch
from ultralytics import YOLO

def train():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training Final Gold Model on: {device} ({torch.cuda.get_device_name(0)})")

    model = YOLO("yolov8s.pt")

    results = model.train(
        data="dataset/yolo_dataset/dataset.yaml",
        epochs=35,
        imgsz=640,
        batch=8,
        device=device,
        workers=2,
        project="cv_core/runs",
        name="rvk_final_gold",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        augment=True,
        mosaic=0.8,
        mixup=0.1,
        close_mosaic=5,
        verbose=True
    )

    best_weights = "cv_core/runs/rvk_final_gold/weights/best.pt"
    if os.path.exists(best_weights):
        import shutil
        shutil.copy2(best_weights, "cv_core/weights/rvk_final_gold.pt")
        print("Copied best weights to cv_core/weights/rvk_final_gold.pt")

if __name__ == "__main__":
    train()
