"""
ppe_prelabel.py - Высокоточная предразметка на специализированной промышленной модели PPE
(melihuzunoglu/ppe-detection: human, helmet, no-helmet, vest) с GPU-ускорением на RTX 4070.
"""

import json
import time
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

def center_in_box(inner_box, container_box):
    ix1, iy1, ix2, iy2 = inner_box
    cx = (ix1 + ix2) / 2
    cy = (iy1 + iy2) / 2
    cx1, cy1, cx2, cy2 = container_box
    return (cx1 <= cx <= cx2) and (cy1 <= cy <= cy2)

def run_ppe_labeling():
    start_time = time.time()
    manifest_path = Path("dataset/splits_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    all_paths = []
    for split in ["train", "val", "test"]:
        all_paths.extend(manifest["splits"][split])

    print(f"=== Промышленная авто-предразметка {len(all_paths)} кадров (PPE Model) ===")

    # Загружаем веса специализированной модели СИЗ
    weights_path = hf_hub_download(repo_id="melihuzunoglu/ppe-detection", filename="best.pt")
    model = YOLO(weights_path)
    print(f"Классы модели: {model.names}")

    crop_dirs = {
        "hardhat": Path("dataset/crops/hardhat"),
        "no_hardhat": Path("dataset/crops/no_hardhat"),
        "safety_vest": Path("dataset/crops/safety_vest"),
        "no_safety_vest": Path("dataset/crops/no_safety_vest"),
    }
    for p in crop_dirs.values():
        p.mkdir(parents=True, exist_ok=True)
        for f in p.glob("*.jpg"):
            f.unlink()

    preview_dir = Path("dataset/previews")
    preview_dir.mkdir(parents=True, exist_ok=True)
    for f in preview_dir.glob("preview_*.jpg"):
        f.unlink()

    coco_data = {
        "info": {
            "description": "RVK Safety Dataset - High-Precision PPE Prelabeling",
            "date_created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": "2.0-PPE"
        },
        "categories": [
            {"id": 1, "name": "person", "supercategory": "human"},
            {"id": 2, "name": "hardhat", "supercategory": "ppe"},
            {"id": 3, "name": "safety_vest", "supercategory": "ppe"}
        ],
        "images": [],
        "annotations": []
    }

    ann_id = 1
    total_humans = 0
    total_helmets = 0
    total_no_helmets = 0
    total_vests = 0
    preview_saved = 0

    for idx, rel_path in enumerate(tqdm(all_paths, desc="Обработка PPE")):
        img_path = Path(rel_path)
        if not img_path.exists():
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        h, w = frame.shape[:2]
        image_id = idx + 1
        coco_data["images"].append({
            "id": image_id,
            "file_name": img_path.name,
            "relative_path": str(img_path),
            "width": w,
            "height": h
        })

        # Инференс модели на GPU
        res = model(frame, conf=0.35, verbose=False)[0]

        humans = []
        helmets = []
        no_helmets = []
        vests = []

        for b in res.boxes:
            cls_name = model.names[int(b.cls.item())]
            conf = float(b.conf.item())
            xyxy = list(map(int, b.xyxy[0].tolist()))
            
            if cls_name == "human":
                bw = xyxy[2] - xyxy[0]
                bh = xyxy[3] - xyxy[1]
                # Фильтр: высота человека не менее 40px и H/W >= 1.3
                if bh >= 40 and (bh / max(bw, 1)) >= 1.3:
                    humans.append((xyxy, conf))
            elif cls_name == "helmet":
                helmets.append((xyxy, conf))
            elif cls_name == "no-helmet":
                no_helmets.append((xyxy, conf))
            elif cls_name == "vest":
                vests.append((xyxy, conf))

        debug_frame = frame.copy() if preview_saved < 25 else None

        for p_idx, (h_box, h_conf) in enumerate(humans):
            total_humans += 1
            hx1, hy1, hx2, hy2 = h_box
            hw = hx2 - hx1
            hh = hy2 - hy1

            # Записываем человека в COCO
            coco_data["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [hx1, hy1, hw, hh],
                "area": hw * hh,
                "score": round(h_conf, 3),
                "iscrowd": 0
            })
            ann_id += 1

            # Вырезаем человека с отступом +15%
            pad_x = int(hw * 0.15)
            pad_y = int(hh * 0.15)
            crop_x1 = max(0, hx1 - pad_x)
            crop_y1 = max(0, hy1 - pad_y)
            crop_x2 = min(w, hx2 + pad_x)
            crop_y2 = min(h, hy2 + pad_y)
            crop_box = [crop_x1, crop_y1, crop_x2, crop_y2]
            crop_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]

            crop_name = f"{img_path.stem}_p{p_idx+1}.jpg"

            # Ищем каску или ее отсутствие внутри этого человека
            has_helmet = False
            for hel_box, hel_conf in helmets:
                if center_in_box(hel_box, crop_box):
                    has_helmet = True
                    total_helmets += 1
                    # COCO hardhat bbox
                    bx1, by1, bx2, by2 = hel_box
                    coco_data["annotations"].append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": 2,
                        "bbox": [bx1, by1, bx2 - bx1, by2 - by1],
                        "area": (bx2 - bx1) * (by2 - by1),
                        "score": round(hel_conf, 3),
                        "iscrowd": 0
                    })
                    ann_id += 1
                    break

            if has_helmet:
                cv2.imwrite(str(crop_dirs["hardhat"] / crop_name), crop_img)
            else:
                total_no_helmets += 1
                cv2.imwrite(str(crop_dirs["no_hardhat"] / crop_name), crop_img)

            # Ищем сигнальный жилет внутри человека
            has_vest = False
            for v_box, v_conf in vests:
                if center_in_box(v_box, crop_box):
                    has_vest = True
                    total_vests += 1
                    vx1, vy1, vx2, vy2 = v_box
                    coco_data["annotations"].append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": 3,
                        "bbox": [vx1, vy1, vx2 - vx1, vy2 - vy1],
                        "area": (vx2 - vx1) * (vy2 - vy1),
                        "score": round(v_conf, 3),
                        "iscrowd": 0
                    })
                    ann_id += 1
                    break

            if has_vest:
                cv2.imwrite(str(crop_dirs["safety_vest"] / crop_name), crop_img)
            else:
                cv2.imwrite(str(crop_dirs["no_safety_vest"] / crop_name), crop_img)

            # Отрисовка на превью
            if debug_frame is not None:
                color = (0, 255, 0) if (has_helmet and has_vest) else ((0, 255, 255) if has_helmet else (0, 0, 255))
                cv2.rectangle(debug_frame, (hx1, hy1), (hx2, hy2), color, 2)
                h_txt = "КАСКА" if has_helmet else "НЕТ КАСКИ"
                v_txt = "ЖИЛЕТ" if has_vest else "БЕЗ ЖИЛЕТА"
                cv2.putText(debug_frame, f"{h_txt} | {v_txt}", (hx1, max(hy1 - 8, 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if debug_frame is not None and len(humans) > 0:
            preview_saved += 1
            cv2.imwrite(str(preview_dir / f"preview_{preview_saved:02d}_{img_path.name}"), debug_frame)

    # Сохраняем итоговый JSON
    out_json = Path("dataset/coco_annotations_draft.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)

    # Обновляем HTML галерею
    preview_imgs = sorted(list(preview_dir.glob("preview_*.jpg")))
    lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>RVK Safety - PPE Previews</title>",
        "<style>body{font-family:sans-serif;background:#0f172a;color:#fff;padding:20px;}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill, minmax(520px, 1fr));gap:20px;}",
        ".card{background:#1e293b;border-radius:10px;overflow:hidden;box-shadow:0 10px 15px -3px rgba(0,0,0,0.5);border:1px solid #334155;}",
        ".card img{width:100%;height:auto;display:block;}",
        ".caption{padding:12px;font-size:14px;color:#94a3b8;font-weight:500;}",
        "h1{color:#38bdf8;margin-bottom:6px;}",
        ".badge{display:inline-block;padding:3px 8px;border-radius:4px;font-size:12px;font-weight:bold;margin-right:8px;}",
        ".badge-green{background:#16a34a;color:#fff;}",
        ".badge-yellow{background:#ca8a04;color:#fff;}",
        ".badge-red{background:#dc2626;color:#fff;}",
        "</style></head><body>",
        "<h1>РВК Safety: Промышленная AI-предразметка (PPE Model)</h1>",
        "<p><span class='badge badge-green'>ЗЕЛЕНЫЙ</span> Каска + Жилет | <span class='badge badge-yellow'>ЖЕЛТЫЙ</span> Каска без жилета | <span class='badge badge-red'>КРАСНЫЙ</span> Нарушение: Нет каски</p>",
        "<div class='grid'>"
    ]
    for im in preview_imgs:
        lines.append(f"<div class='card'><img src='{im.name}'><div class='caption'>{im.name}</div></div>")
    lines.append("</div></body></html>")
    (preview_dir / "gallery.html").write_text("\n".join(lines), encoding="utf-8")

    elapsed = round(time.time() - start_time, 2)
    print(f"\n=== Успех! Промышленная модель завершила разметку за {elapsed} сек ===")
    print(f"Людей (human):           {total_humans}")
    print(f"В касках (helmet):       {total_helmets}")
    print(f"Без каски (no-helmet):   {total_no_helmets}")
    print(f"В сигнальных жилетах:    {total_vests}")
    print(f"Превью-галерея: dataset/previews/gallery.html")

if __name__ == "__main__":
    run_ppe_labeling()
