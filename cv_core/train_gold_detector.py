import os
import torch
from ultralytics import YOLO

def train():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training Gold Model on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Start from baseline weights
    model = YOLO("yolov8s.pt")

    print("\n--- Training Gold Detector on Rosvodokanal CCTV Scenes (30 Epochs) ---")
    results = model.train(
        data="dataset/yolo_dataset/dataset.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        device=device,
        workers=2,
        project="cv_core/runs",
        name="rvk_gold_detector",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        augment=True,
        mosaic=1.0,
        mixup=0.15,
        close_mosaic=5,
        verbose=True
    )

    best_weights = "cv_core/runs/rvk_gold_detector/weights/best.pt"
    if os.path.exists(best_weights):
        import shutil
        shutil.copy2(best_weights, "cv_core/weights/rvk_gold_detector.pt")
        print(f"\nSuccessfully copied best weights to cv_core/weights/rvk_gold_detector.pt")

if __name__ == "__main__":
    train()
