import os
import sys
import time
import glob
import json
import random
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def synthesize_safety_vest(img):
    img = img.copy()
    w, h = img.size
    torso_top = int(h * 0.22)
    torso_bot = int(h * 0.72)
    if torso_bot <= torso_top + 10 or w < 15:
        return img
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    vest_color = (255, 95, 0, 185) if random.random() < 0.5 else (195, 235, 10, 185)
    stripe_color = (230, 235, 245, 220)
    pad_x = int(w * 0.08)
    draw.rectangle([pad_x, torso_top, w - pad_x, torso_bot], fill=vest_color)
    s1_y = torso_top + int((torso_bot - torso_top) * 0.45)
    s2_y = torso_top + int((torso_bot - torso_top) * 0.75)
    s_thick = max(2, int(h * 0.03))
    draw.rectangle([pad_x, s1_y, w - pad_x, s1_y + s_thick], fill=stripe_color)
    draw.rectangle([pad_x, s2_y, w - pad_x, s2_y + s_thick], fill=stripe_color)
    img = img.convert("RGBA")
    return Image.alpha_composite(img, overlay).convert("RGB")

class PPEDataset(Dataset):
    def __init__(self, root_dir, is_train=True, transform=None):
        self.samples = []
        self.is_train = is_train
        self.transform = transform
        
        hh_paths = glob.glob(os.path.join(root_dir, "hardhat", "*.jpg"))
        nohh_paths = glob.glob(os.path.join(root_dir, "no_hardhat", "*.jpg"))
        
        for p in hh_paths:
            self.samples.append((p, 1, 0, False))
        for p in nohh_paths:
            self.samples.append((p, 0, 0, False))
            
        # Synthesize safety vests for both training and validation
        random.seed(42 if not self.is_train else int(time.time()))
        num_vest_samples = int(len(self.samples) * 0.50)
        selected = random.sample(self.samples, num_vest_samples) if self.samples else []
        for p, hh_label, _, _ in selected:
            self.samples.append((p, hh_label, 1, True))
                
        random.seed(42)
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, hh_label, vest_label, is_synthetic = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if is_synthetic:
            img = synthesize_safety_vest(img)
            
        w, h = img.size
        head_h = max(int(h * 0.35), 20)
        head_img = img.crop((0, 0, w, min(head_h, h)))
        
        if self.transform:
            head_tensor = self.transform(head_img)
            body_tensor = self.transform(img)
        else:
            to_tensor = transforms.ToTensor()
            head_tensor = to_tensor(head_img)
            body_tensor = to_tensor(img)
            
        return head_tensor, body_tensor, torch.tensor(hh_label, dtype=torch.float32), torch.tensor(vest_label, dtype=torch.float32)

# --- 1. MobileNetV3-Small ---
class MobileNetV3Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = self.head_backbone.classifier[0].in_features
        self.head_backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1)
        )
        self.body_backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_b = self.body_backbone.classifier[0].in_features
        self.body_backbone.classifier = nn.Sequential(
            nn.Linear(in_b, 64),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1)
        )

    def forward(self, head_img, body_img):
        hh = self.head_backbone(head_img).squeeze(-1)
        vest = self.body_backbone(body_img).squeeze(-1)
        return hh, vest

# --- 2. EfficientNet-B0 (Google Brain SOTA) ---
class EfficientNetB0Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = self.head_backbone.classifier[1].in_features
        self.head_backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        self.body_backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_b = self.body_backbone.classifier[1].in_features
        self.body_backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_b, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, head_img, body_img):
        hh = self.head_backbone(head_img).squeeze(-1)
        vest = self.body_backbone(body_img).squeeze(-1)
        return hh, vest

# --- 3. ResNet-18 (Deep Residual Baseline) ---
class ResNet18Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.head_backbone.fc.in_features
        self.head_backbone.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.body_backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_b = self.body_backbone.fc.in_features
        self.body_backbone.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_b, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, head_img, body_img):
        hh = self.head_backbone(head_img).squeeze(-1)
        vest = self.body_backbone(body_img).squeeze(-1)
        return hh, vest

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_and_eval_architecture(model_name: str, model: nn.Module, train_loader, val_loader, epochs=12):
    print(f"\n==========================================")
    print(f"[BENCHMARK] Training & Evaluating: {model_name} on {DEVICE}")
    print(f"==========================================")
    
    model = model.to(DEVICE)
    hh_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.0]).to(DEVICE))
    vest_criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_f1 = 0.0
    best_weights_path = os.path.join(BASE_DIR, "cv_core", "weights", f"{model_name.lower().replace('-', '_')}_best.pth")
    
    start_train_t = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for heads, bodies, hh_labels, vest_labels in train_loader:
            heads = heads.to(DEVICE)
            bodies = bodies.to(DEVICE)
            hh_labels = hh_labels.to(DEVICE)
            vest_labels = vest_labels.to(DEVICE)
            
            optimizer.zero_grad()
            hh_preds, vest_preds = model(heads, bodies)
            loss = hh_criterion(hh_preds, hh_labels) + vest_criterion(vest_preds, vest_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(hh_labels)
            
        scheduler.step()
        
        # Validation
        model.eval()
        all_hh_preds, all_hh_targets = [], []
        all_vest_preds, all_vest_targets = [], []
        with torch.no_grad():
            for heads, bodies, hh_labels, vest_labels in val_loader:
                heads = heads.to(DEVICE)
                bodies = bodies.to(DEVICE)
                hh_preds, vest_preds = model(heads, bodies)
                hh_probs = torch.sigmoid(hh_preds).cpu().numpy()
                vest_probs = torch.sigmoid(vest_preds).cpu().numpy()
                all_hh_preds.extend((hh_probs >= 0.5).astype(int))
                all_hh_targets.extend(hh_labels.numpy().astype(int))
                all_vest_preds.extend((vest_probs >= 0.5).astype(int))
                all_vest_targets.extend(vest_labels.numpy().astype(int))
                
        all_hh_preds = np.array(all_hh_preds)
        all_hh_targets = np.array(all_hh_targets)
        tp = np.sum((all_hh_preds == 1) & (all_hh_targets == 1))
        fp = np.sum((all_hh_preds == 1) & (all_hh_targets == 0))
        fn = np.sum((all_hh_preds == 0) & (all_hh_targets == 1))
        tn = np.sum((all_hh_preds == 0) & (all_hh_targets == 0))
        
        acc = (tp + tn) / max(len(all_hh_targets), 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1_hh = 2 * prec * rec / max(prec + rec, 1e-6)

        all_vest_preds = np.array(all_vest_preds)
        all_vest_targets = np.array(all_vest_targets)
        tp_v = np.sum((all_vest_preds == 1) & (all_vest_targets == 1))
        fp_v = np.sum((all_vest_preds == 1) & (all_vest_targets == 0))
        fn_v = np.sum((all_vest_preds == 0) & (all_vest_targets == 1))
        tn_v = np.sum((all_vest_preds == 0) & (all_vest_targets == 0))
        prec_v = tp_v / max(tp_v + fp_v, 1)
        rec_v = tp_v / max(tp_v + fn_v, 1)
        f1_vest = 2 * prec_v * rec_v / max(prec_v + rec_v, 1e-6)

        f1 = (f1_hh + f1_vest) / 2.0
        
        if f1 > best_f1 or epoch == 1:
            best_f1 = f1
            torch.save(model.state_dict(), best_weights_path)
            
        if epoch % 3 == 0 or epoch == epochs:
            print(f"[{model_name}] Epoch {epoch:02d}/{epochs:02d} | HH F1: {f1_hh:.3f} | Vest F1: {f1_vest:.3f} | Combined F1: {f1:.3f}")

    train_duration = time.time() - start_train_t
    
    # Load best weights for final benchmark
    model.load_state_dict(torch.load(best_weights_path, weights_only=True))
    model.eval()
    
    # Latency benchmark: 50 warmups + 100 runs
    dummy_head = torch.randn(1, 3, 128, 128).to(DEVICE)
    dummy_body = torch.randn(1, 3, 128, 128).to(DEVICE)
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_head, dummy_body)
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        iters = 100
        for _ in range(iters):
            _ = model(dummy_head, dummy_body)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        gpu_latency_ms = ((t1 - t0) / iters) * 1000.0

    # CPU Latency benchmark
    model_cpu = model.to("cpu")
    dummy_head_cpu = dummy_head.to("cpu")
    dummy_body_cpu = dummy_body.to("cpu")
    with torch.no_grad():
        for _ in range(5):
            _ = model_cpu(dummy_head_cpu, dummy_body_cpu)
        t0 = time.perf_counter()
        iters_cpu = 30
        for _ in range(iters_cpu):
            _ = model_cpu(dummy_head_cpu, dummy_body_cpu)
        t1 = time.perf_counter()
        cpu_latency_ms = ((t1 - t0) / iters_cpu) * 1000.0

    # Full Evaluation on Val for both Hardhat and Vest
    all_hh_preds, all_hh_targets = [], []
    all_vest_preds, all_vest_targets = [], []
    with torch.no_grad():
        for heads, bodies, hh_labels, vest_labels in val_loader:
            heads = heads.to("cpu")
            bodies = bodies.to("cpu")
            hh_preds, vest_preds = model_cpu(heads, bodies)
            hh_probs = torch.sigmoid(hh_preds).numpy()
            vest_probs = torch.sigmoid(vest_preds).numpy()
            all_hh_preds.extend((hh_probs >= 0.5).astype(int))
            all_hh_targets.extend(hh_labels.numpy().astype(int))
            all_vest_preds.extend((vest_probs >= 0.5).astype(int))
            all_vest_targets.extend(vest_labels.numpy().astype(int))

    all_hh_preds = np.array(all_hh_preds)
    all_hh_targets = np.array(all_hh_targets)
    tp = int(np.sum((all_hh_preds == 1) & (all_hh_targets == 1)))
    fp = int(np.sum((all_hh_preds == 1) & (all_hh_targets == 0)))
    fn = int(np.sum((all_hh_preds == 0) & (all_hh_targets == 1)))
    tn = int(np.sum((all_hh_preds == 0) & (all_hh_targets == 0)))

    acc = float((tp + tn) / max(len(all_hh_targets), 1))
    prec = float(tp / max(tp + fp, 1))
    rec = float(tp / max(tp + fn, 1))
    f1_hh = float(2 * prec * rec / max(prec + rec, 1e-6))

    all_vest_preds = np.array(all_vest_preds)
    all_vest_targets = np.array(all_vest_targets)
    tp_v = int(np.sum((all_vest_preds == 1) & (all_vest_targets == 1)))
    fp_v = int(np.sum((all_vest_preds == 1) & (all_vest_targets == 0)))
    fn_v = int(np.sum((all_vest_preds == 0) & (all_vest_targets == 1)))
    tn_v = int(np.sum((all_vest_preds == 0) & (all_vest_targets == 0)))
    prec_v = float(tp_v / max(tp_v + fp_v, 1))
    rec_v = float(tp_v / max(tp_v + fn_v, 1))
    f1_vest = float(2 * prec_v * rec_v / max(prec_v + rec_v, 1e-6))
    f1_combined = (f1_hh + f1_vest) / 2.0

    params_m = round(count_parameters(model) / 1e6, 2)
    filesize_mb = round(os.path.getsize(best_weights_path) / (1024 * 1024), 2)

    result = {
        "model_name": model_name,
        "weights_path": best_weights_path,
        "params_millions": params_m,
        "weights_size_mb": filesize_mb,
        "accuracy": round(acc * 100, 1),
        "precision": round(prec * 100, 1),
        "recall": round(rec * 100, 1),
        "f1_score": round(f1_combined, 3),
        "hardhat_f1": round(f1_hh, 3),
        "vest_f1": round(f1_vest, 3),
        "gpu_latency_ms": round(gpu_latency_ms, 2),
        "cpu_latency_ms": round(cpu_latency_ms, 2),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "vest_confusion_matrix": {"tp": tp_v, "fp": fp_v, "fn": fn_v, "tn": tn_v},
        "training_time_sec": round(train_duration, 1)
    }
    return result

def run_all_benchmarks():
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_ds = PPEDataset(os.path.join(BASE_DIR, "dataset", "clean", "train"), is_train=True, transform=transform)
    val_ds = PPEDataset(os.path.join(BASE_DIR, "dataset", "clean", "val"), is_train=False, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    
    print(f"Data: Train={len(train_ds)} crops, Val={len(val_ds)} crops")

    models_to_test = [
        ("MobileNetV3-Small", MobileNetV3Classifier()),
        ("EfficientNet-B0", EfficientNetB0Classifier()),
        ("ResNet-18", ResNet18Classifier())
    ]
    
    benchmark_results = []
    for name, m in models_to_test:
        res = train_and_eval_architecture(name, m, train_loader, val_loader, epochs=10)
        benchmark_results.append(res)
        
    out_json = os.path.join(BASE_DIR, "backend", "data", "models_benchmark.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, ensure_ascii=False, indent=2)
        
    print("\n================ BENCHMARK SUMMARY ================")
    for r in benchmark_results:
        print(f"* {r['model_name']:<18} | Acc: {r['accuracy']}% | F1: {r['f1_score']} | Prec: {r['precision']}% | Rec: {r['recall']}% | GPU: {r['gpu_latency_ms']}ms | CPU: {r['cpu_latency_ms']}ms | Params: {r['params_millions']}M")
    print(f"Saved results to {out_json}")
    return benchmark_results

if __name__ == "__main__":
    run_all_benchmarks()
