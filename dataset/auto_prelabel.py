"""
auto_prelabel.py - Автоматическая предразметка с жесткими фильтрами станков
и точной калибровкой на данных «РВК Safety».
"""

import json
import time
import cv2
import numpy as np
import torch
import torchvision
from pathlib import Path
from tqdm import tqdm

def analyze_head_hardhat(person_crop: np.ndarray) -> tuple[bool, float, list[int]]:
    h, w, _ = person_crop.shape
    if h < 25 or w < 12:
        return False, 0.0, [0, 0, 0, 0]
        
    head_h = max(int(h * 0.22), 8)
    cx1 = int(w * 0.15)
    cx2 = int(w * 0.85)
    head_roi = person_crop[0:head_h, cx1:cx2]
    if head_roi.size == 0:
        return False, 0.0, [0, 0, 0, 0]
    
    hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
    h_c, s_c, v_c = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    
    # Строгий поиск цветов касок (оранжевый, желтый, синий, ярко-белый купол)
    yellow_orange = (h_c >= 12) & (h_c <= 38) & (s_c > 110) & (v_c > 120)
    bright_blue = (h_c >= 95) & (h_c <= 125) & (s_c > 110) & (v_c > 90)
    pure_white_dome = (v_c > 220) & (s_c < 30)
    
    helmet_pixels = yellow_orange | bright_blue | pure_white_dome
    ratio = np.sum(helmet_pixels) / float(head_roi.shape[0] * head_roi.shape[1])
    
    # Требуется существенное покрытие характерным цветом каски
    has_hardhat = bool(ratio > 0.22)
    conf = min(0.6 + ratio * 0.8, 0.98) if has_hardhat else max(0.92 - ratio * 2.5, 0.5)
    
    hw = max(cx2 - cx1, 5)
    hh = int(head_h * 0.9)
    return has_hardhat, float(conf), [cx1, 0, hw, hh]

def analyze_torso_vest(person_crop: np.ndarray) -> tuple[bool, float, list[int]]:
    h, w, _ = person_crop.shape
    if h < 35 or w < 15:
        return False, 0.0, [0, 0, 0, 0]
        
    y_start = int(h * 0.22)
    y_end = int(h * 0.65)
    torso_roi = person_crop[y_start:y_end, :]
    if torso_roi.size == 0:
        return False, 0.0, [0, 0, 0, 0]
        
    hsv = cv2.cvtColor(torso_roi, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    
    # Флуоресцентный оранжевый или салатовый
    fluorescent_orange = (h_chan >= 5) & (h_chan <= 25) & (s_chan > 140) & (v_chan > 140)
    fluorescent_green = (h_chan >= 30) & (h_chan <= 45) & (s_chan > 130) & (v_chan > 130)
    reflective_stripes = (v_chan > 235) & (s_chan < 50)
    
    vest_mask = fluorescent_orange | fluorescent_green | reflective_stripes
    ratio = np.sum(vest_mask) / float(torso_roi.shape[0] * torso_roi.shape[1])
    
    has_vest = bool(ratio > 0.16)
    conf = min(0.6 + ratio * 0.9, 0.96) if has_vest else max(0.93 - ratio * 3, 0.5)
    
    vx1 = int(w * 0.08)
    vy1 = y_start
    vw = int(w * 0.84)
    vh = y_end - y_start
    return has_vest, float(conf), [vx1, vy1, vw, vh]

def run_prelabeling():
    start_time = time.time()
    manifest_path = Path("dataset/splits_manifest.json")
    if not manifest_path.exists():
        print("Ошибка: манифест dataset/splits_manifest.json не найден!")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    all_unique_paths = []
    for split_name in ["train", "val", "test"]:
        all_unique_paths.extend(manifest["splits"][split_name])

    print(f"=== Запуск калиброванной авто-предразметки {len(all_unique_paths)} кадров ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    model.to(device)
    model.eval()

    crop_dirs = {
        "hardhat": Path("dataset/crops/hardhat"),
        "no_hardhat": Path("dataset/crops/no_hardhat"),
        "safety_vest": Path("dataset/crops/safety_vest"),
        "no_safety_vest": Path("dataset/crops/no_safety_vest"),
    }
    for p in crop_dirs.values():
        p.mkdir(parents=True, exist_ok=True)
        # Очистим старые файлы перед чистым прогоном
        for f in p.glob("*.jpg"):
            f.unlink()

    preview_dir = Path("dataset/previews")
    preview_dir.mkdir(parents=True, exist_ok=True)
    for f in preview_dir.glob("preview_*.jpg"):
        f.unlink()

    coco_data = {
        "info": {
            "description": "RVK Safety Dataset - Calibrated Draft",
            "date_created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": "1.1"
        },
        "categories": [
            {"id": 1, "name": "person", "supercategory": "human"},
            {"id": 2, "name": "hardhat", "supercategory": "ppe"},
            {"id": 3, "name": "safety_vest", "supercategory": "ppe"}
        ],
        "images": [],
        "annotations": []
    }

    ann_id_counter = 1
    total_persons = 0
    total_hardhats = 0
    total_vests = 0
    preview_count = 0

    for img_idx, rel_path in enumerate(tqdm(all_unique_paths, desc="Калиброванная обработка")):
        img_path = Path(rel_path)
        if not img_path.exists():
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        orig_h, orig_w = frame.shape[:2]
        image_id = img_idx + 1

        coco_data["images"].append({
            "id": image_id,
            "file_name": img_path.name,
            "relative_path": str(img_path),
            "width": orig_w,
            "height": orig_h
        })

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor_img = torch.from_numpy(rgb_frame).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(tensor_img)[0]

        boxes = preds["boxes"].cpu().numpy()
        labels = preds["labels"].cpu().numpy()
        scores = preds["scores"].cpu().numpy()

        # ЖЕСТКИЙ ФИЛЬТР: только человек, score >= 0.72 (отсекает синие станки и трубы)
        person_mask = (labels == 1) & (scores >= 0.72)
        person_boxes = boxes[person_mask]
        person_scores = scores[person_mask]

        debug_frame = frame.copy() if preview_count < 25 else None
        has_valid_person_in_frame = False

        for p_idx, (p_box, p_score) in enumerate(zip(person_boxes, person_scores)):
            x1, y1, x2, y2 = map(int, p_box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)
            bw = x2 - x1
            bh = y2 - y1

            # Фильтр пропорций: человек должен быть вертикальным (H/W >= 1.45) и высотой >= 45 px
            if bw < 18 or bh < 45:
                continue
            if (bh / float(bw)) < 1.45:
                continue # Станки, краны и квадратные трубы отсеиваются!

            has_valid_person_in_frame = True
            total_persons += 1

            coco_data["annotations"].append({
                "id": ann_id_counter,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x1, y1, bw, bh],
                "area": bw * bh,
                "score": float(p_score),
                "iscrowd": 0
            })
            ann_id_counter += 1

            pad_x = int(bw * 0.15)
            pad_y = int(bh * 0.15)
            crop_x1 = max(0, x1 - pad_x)
            crop_y1 = max(0, y1 - pad_y)
            crop_x2 = min(orig_w, x2 + pad_x)
            crop_y2 = min(orig_h, y2 + pad_y)
            person_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

            has_hardhat, h_conf, h_local = analyze_head_hardhat(person_crop)
            has_vest, v_conf, v_local = analyze_torso_vest(person_crop)

            crop_filename = f"{img_path.stem}_p{p_idx+1}.jpg"

            if has_hardhat:
                total_hardhats += 1
                cv2.imwrite(str(crop_dirs["hardhat"] / crop_filename), person_crop)
                hx = crop_x1 + h_local[0]
                hy = crop_y1 + h_local[1]
                coco_data["annotations"].append({
                    "id": ann_id_counter,
                    "image_id": image_id,
                    "category_id": 2,
                    "bbox": [hx, hy, h_local[2], h_local[3]],
                    "area": h_local[2] * h_local[3],
                    "score": h_conf,
                    "iscrowd": 0
                })
                ann_id_counter += 1
            else:
                cv2.imwrite(str(crop_dirs["no_hardhat"] / crop_filename), person_crop)

            if has_vest:
                total_vests += 1
                cv2.imwrite(str(crop_dirs["safety_vest"] / crop_filename), person_crop)
                vx = crop_x1 + v_local[0]
                vy = crop_y1 + v_local[1]
                coco_data["annotations"].append({
                    "id": ann_id_counter,
                    "image_id": image_id,
                    "category_id": 3,
                    "bbox": [vx, vy, v_local[2], v_local[3]],
                    "area": v_local[2] * v_local[3],
                    "score": v_conf,
                    "iscrowd": 0
                })
                ann_id_counter += 1
            else:
                cv2.imwrite(str(crop_dirs["no_safety_vest"] / crop_filename), person_crop)

            if debug_frame is not None:
                color = (0, 255, 0) if (has_hardhat and has_vest) else ((0, 255, 255) if has_hardhat else (0, 0, 255))
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
                h_text = "H+" if has_hardhat else "H-"
                v_text = "V+" if has_vest else "V-"
                cv2.putText(debug_frame, f"{h_text} {v_text} ({p_score:.2f})", (x1, max(y1 - 6, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if debug_frame is not None and has_valid_person_in_frame:
            preview_count += 1
            cv2.imwrite(str(preview_dir / f"preview_{preview_count:02d}_{img_path.name}"), debug_frame)

    coco_output_path = Path("dataset/coco_annotations_draft.json")
    with open(coco_output_path, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)

    # Обновляем HTML галерею
    preview_files = sorted(list(preview_dir.glob("preview_*.jpg")))
    lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>RVK Safety - Previews</title>",
        "<style>body{font-family:sans-serif;background:#141414;color:#fff;padding:20px;}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill, minmax(500px, 1fr));gap:20px;}",
        ".card{background:#222;border-radius:8px;overflow:hidden;border:1px solid #333;}",
        ".card img{width:100%;height:auto;display:block;}",
        ".caption{padding:10px;font-size:13px;color:#888;}",
        "h1{color:#38bdf8;}</style></head><body>",
        "<h1>РВК Safety: Калиброванная предразметка (RTX 4070)</h1>",
        "<p>Фильтры: Score &ge; 0.72, Пропорции H/W &ge; 1.45 (станки отсеяны). Зеленый: Каска+Жилет | Желтый: Каска без жилета | Красный: Без каски</p>",
        "<div class=\"grid\">"
    ]
    for img in preview_files:
        lines.append(f"<div class=\"card\"><img src=\"{img.name}\"><div class=\"caption\">{img.name}</div></div>")
    lines.append("</div></body></html>")
    (preview_dir / "gallery.html").write_text("\n".join(lines), encoding="utf-8")

    elapsed = round(time.time() - start_time, 2)
    print(f"\nКалибровка завершена за {elapsed} сек!")
    print(f"Честных людей найдено: {total_persons}")
    print(f"Касок: {total_hardhats} | Жилетов: {total_vests}")
    print(f"Галерея обновлена: dataset/previews/gallery.html")

if __name__ == "__main__":
    run_prelabeling()
