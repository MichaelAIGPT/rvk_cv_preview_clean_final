import os
import torch
from ultralytics import YOLO

def train():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load baseline model
    # We use yolov8s.pt or transfer weights from ppe_detector_best.pt
    model = YOLO("yolov8s.pt")

    print("\n--- Starting End-to-End Fine-Tuning on Rosvodokanal CCTV Scenes (25 Epochs) ---")
    results = model.train(
        data="dataset/yolo_dataset/dataset.yaml",
        epochs=25,
        imgsz=640,
        batch=8,
        device=device,
        workers=2,
        project="cv_core/runs",
        name="rvk_detector_ft",
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

    best_weights = "cv_core/runs/rvk_detector_ft/weights/best.pt"
    if os.path.exists(best_weights):
        import shutil
        shutil.copy2(best_weights, "cv_core/weights/rvk_finetuned_detector.pt")
        print(f"\nSuccessfully copied best weights to cv_core/weights/rvk_finetuned_detector.pt")

if __name__ == "__main__":
    train()
