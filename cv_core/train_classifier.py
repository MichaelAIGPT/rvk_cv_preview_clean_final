import os
import sys
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

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using training device: {DEVICE}")

# 1. TRIZ Vest Synthesis: Augment torso with realistic high-vis vest
def synthesize_safety_vest(img):
    img = img.copy()
    w, h = img.size
    torso_top = int(h * 0.22)
    torso_bot = int(h * 0.72)
    
    if torso_bot <= torso_top + 10 or w < 15:
        return img
        
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # 50% orange vest, 50% lime vest
    if random.random() < 0.5:
        vest_color = (255, 95, 0, 185) # Safety orange
    else:
        vest_color = (195, 235, 10, 185) # High-vis lime
        
    stripe_color = (230, 235, 245, 220) # Silver reflective stripe
    
    pad_x = int(w * 0.08)
    draw.rectangle([pad_x, torso_top, w - pad_x, torso_bot], fill=vest_color)
    
    s1_y = torso_top + int((torso_bot - torso_top) * 0.45)
    s2_y = torso_top + int((torso_bot - torso_top) * 0.75)
    s_thick = max(2, int(h * 0.03))
    draw.rectangle([pad_x, s1_y, w - pad_x, s1_y + s_thick], fill=stripe_color)
    draw.rectangle([pad_x, s2_y, w - pad_x, s2_y + s_thick], fill=stripe_color)
    
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay).convert("RGB")
    return img

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
            
        if self.is_train:
            num_vest_samples = int(len(self.samples) * 0.45)
            selected = random.sample(self.samples, num_vest_samples)
            for p, hh_label, _, _ in selected:
                self.samples.append((p, hh_label, 1, True))
                
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

class MultiTaskPPEClassifier(nn.Module):
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
        in_features_body = self.body_backbone.classifier[0].in_features
        self.body_backbone.classifier = nn.Sequential(
            nn.Linear(in_features_body, 64),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1)
        )

    def forward(self, head_img, body_img):
        hh_logit = self.head_backbone(head_img).squeeze(-1)
        vest_logit = self.body_backbone(body_img).squeeze(-1)
        return hh_logit, vest_logit

def train():
    os.makedirs("cv_core/weights", exist_ok=True)
    
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_ds = PPEDataset("dataset/clean/train", is_train=True, transform=train_transform)
    val_ds = PPEDataset("dataset/clean/val", is_train=False, transform=val_transform)
    
    print(f"Dataset loaded: Train={len(train_ds)} samples, Val={len(val_ds)} samples")
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    
    model = MultiTaskPPEClassifier().to(DEVICE)
    
    hh_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.2]).to(DEVICE))
    vest_criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    best_f1 = 0.0
    best_epoch = 0
    epochs = 20
    
    print("\n--- Training on RTX 4070 (20 Epochs) ---")
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
            
            loss_hh = hh_criterion(hh_preds, hh_labels)
            loss_vest = vest_criterion(vest_preds, vest_labels)
            loss = loss_hh + loss_vest
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(hh_labels)
            
        scheduler.step()
        train_loss /= len(train_ds)
        
        model.eval()
        all_hh_preds = []
        all_hh_targets = []
        
        with torch.no_grad():
            for heads, bodies, hh_labels, vest_labels in val_loader:
                heads = heads.to(DEVICE)
                bodies = bodies.to(DEVICE)
                hh_preds, _ = model(heads, bodies)
                probs = torch.sigmoid(hh_preds).cpu().numpy()
                all_hh_preds.extend((probs >= 0.5).astype(int))
                all_hh_targets.extend(hh_labels.numpy().astype(int))
                
        all_hh_preds = np.array(all_hh_preds)
        all_hh_targets = np.array(all_hh_targets)
        
        tp = np.sum((all_hh_preds == 1) & (all_hh_targets == 1))
        fp = np.sum((all_hh_preds == 1) & (all_hh_targets == 0))
        fn = np.sum((all_hh_preds == 0) & (all_hh_targets == 1))
        tn = np.sum((all_hh_preds == 0) & (all_hh_targets == 0))
        
        acc = (tp + tn) / max(len(all_hh_targets), 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-6)
        
        if epoch % 2 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Acc: {acc*100:.1f}% | Rec: {rec*100:.1f}% | F1: {f1:.3f} (TP={tp}, FP={fp}, FN={fn}, TN={tn})")
            
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            torch.save(model.state_dict(), "cv_core/weights/best_ppe_classifier.pth")

    print(f"\nTraining Complete! Best Validation F1: {best_f1:.3f} at Epoch {best_epoch}")
    torch.save(model.state_dict(), "cv_core/weights/ppe_classifier_final.pth")
    print("PyTorch model saved to cv_core/weights/ppe_classifier_final.pth")

if __name__ == "__main__":
    train()
