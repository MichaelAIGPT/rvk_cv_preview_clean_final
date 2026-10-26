import os
import json
import time
import uuid
import cv2
from datetime import datetime
from typing import List, Dict, Optional, Union, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE_DIR = os.path.join(BASE_DIR, "backend", "data", "active_learning_store")
CROPS_DIR = os.path.join(STORE_DIR, "crops")
FRAMES_DIR = os.path.join(STORE_DIR, "frames")
MANIFEST_FILE = os.path.join(STORE_DIR, "manifest.json")

os.makedirs(CROPS_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)


class ActiveLearningManager:
    """Manages physical storage of active learning crops, full frames, and annotations."""

    def __init__(self):
        self._init_manifest()

    def _init_manifest(self):
        if not os.path.exists(MANIFEST_FILE):
            with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)

    def get_manifest(self) -> List[Dict]:
        try:
            with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def get_all_samples(self) -> List[Dict]:
        return self.get_manifest()

    def _save_manifest(self, data: List[Dict]):
        with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _calc_iou(b1, b2) -> float:
        xA = max(b1[0], b2[0])
        yA = max(b1[1], b2[1])
        xB = min(b1[2], b2[2])
        yB = min(b1[3], b2[3])
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        area1 = max(1, (b1[2] - b1[0]) * (b1[3] - b1[1]))
        area2 = max(1, (b2[2] - b2[0]) * (b2[3] - b2[1]))
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def save_sample(
        self,
        frame_bgr,
        source_frame_name: str,
        playlist: str,
        bbox: List[int],
        user_label: str,
        track_id: Optional[str] = None,
        model_prediction: Optional[Dict] = None
    ) -> Dict:
        """
        Crops the object from frame_bgr with 10% context margin,
        saves full frame and crop to disk, and updates manifest.
        Deduplicates prior conflicting annotations on the same frame / bbox.
        """
        sample_id = f"al-{uuid.uuid4().hex[:8]}"
        now = datetime.now()
        timestamp = time.time()
        time_str = now.strftime("%Y-%m-%d %H:%M:%S")

        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Bounding box sanity check
        x1 = max(0, min(w - 1, x1))
        x2 = max(x1 + 1, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(y1 + 1, min(h, y2))

        # Add 10% margin for context crop
        bw = x2 - x1
        bh = y2 - y1
        pad_x = int(bw * 0.10)
        pad_y = int(bh * 0.10)
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(w, x2 + pad_x)
        crop_y2 = min(h, y2 + pad_y)

        crop_bgr = frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2]

        crop_filename = f"{sample_id}_{user_label}.jpg"
        crop_rel_path = f"crops/{crop_filename}"
        crop_full_path = os.path.join(CROPS_DIR, crop_filename)
        cv2.imwrite(crop_full_path, crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        frame_filename = f"{sample_id}_full.jpg"
        frame_rel_path = f"frames/{frame_filename}"
        frame_full_path = os.path.join(FRAMES_DIR, frame_filename)
        cv2.imwrite(frame_full_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Category mapping
        category_map = {
            "hardhat": "В каске (Рабочий)",
            "no_hardhat": "Без каски (Нарушитель)",
            "not_person_pump": "⚙️ Оборудование (Насосы / Приводы)",
            "not_person_pipe": "🔵 Трубопроводы (Трубы / Задвижки)",
            "not_person_stand": "🏗️ Стенд / Металлоконструкция",
            "not_person": "🚫 Фоновый артефакт (Не человек)",
            "multiple_people": "👥 Склейка (Группа)"
        }
        category_title = category_map.get(user_label, user_label)

        sample_record = {
            "id": sample_id,
            "timestamp": timestamp,
            "time_str": time_str,
            "source_frame": source_frame_name,
            "playlist": playlist,
            "track_id": str(track_id) if track_id is not None else "manual",
            "bbox": [x1, y1, x2, y2],
            "crop_coords": [crop_x1, crop_y1, crop_x2, crop_y2],
            "crop_url": f"/learning_store/{crop_rel_path}",
            "frame_url": f"/learning_store/{frame_rel_path}",
            "user_label": user_label,
            "category_title": category_title,
            "is_person": user_label in ("hardhat", "no_hardhat", "multiple_people"),
            "batch": "v3_active_mining",
            "stage": "pending_v3",
            "model_prediction": model_prediction or {
                "has_hardhat": False,
                "confidence": 0.0
            }
        }

        # Deduplicate prior conflicting samples for the same frame/bbox or manual track
        manifest = self.get_manifest()
        current_bbox = [x1, y1, x2, y2]
        clean_manifest = []
        for s in manifest:
            same_frame = (s.get("source_frame") == source_frame_name)
            s_tid = str(s.get("track_id", ""))
            curr_tid = str(track_id or "")
            same_manual_track = (curr_tid and s_tid == curr_tid and not curr_tid.startswith("auto-"))
            is_overlap = same_frame and self._calc_iou(s.get("bbox", [0, 0, 0, 0]), current_bbox) > 0.45

            if same_manual_track or is_overlap:
                # Remove obsolete crop/frame if they exist
                crop_path = os.path.join(STORE_DIR, s.get("crop_url", "").replace("/learning_store/", "").replace("/", os.sep))
                frame_path = os.path.join(STORE_DIR, s.get("frame_url", "").replace("/learning_store/", "").replace("/", os.sep))
                if os.path.exists(crop_path):
                    try:
                        os.remove(crop_path)
                    except Exception:
                        pass
                if os.path.exists(frame_path):
                    try:
                        os.remove(frame_path)
                    except Exception:
                        pass
            else:
                clean_manifest.append(s)

        clean_manifest.insert(0, sample_record)  # Newest first
        self._save_manifest(clean_manifest)

        return sample_record

    def delete_sample(self, sample_id: str) -> bool:
        manifest = self.get_manifest()
        target = None
        remaining = []
        for s in manifest:
            if s.get("id") == sample_id:
                target = s
            else:
                remaining.append(s)

        if target:
            # Delete physical files
            crop_path = os.path.join(STORE_DIR, target.get("crop_url", "").replace("/learning_store/", "").replace("/", os.sep))
            frame_path = os.path.join(STORE_DIR, target.get("frame_url", "").replace("/learning_store/", "").replace("/", os.sep))
            if os.path.exists(crop_path):
                try:
                    os.remove(crop_path)
                except Exception:
                    pass
            if os.path.exists(frame_path):
                try:
                    os.remove(frame_path)
                except Exception:
                    pass

            self._save_manifest(remaining)
            return True
        return False

    def get_stats(self) -> Dict:
        manifest = self.get_manifest()
        stats = {
            "total": len(manifest),
            "hardhat": 0,
            "no_hardhat": 0,
            "not_person_pump": 0,
            "not_person_pipe": 0,
            "not_person_stand": 0,
            "not_person_other": 0,
            "multiple_people": 0
        }
        for item in manifest:
            lbl = item.get("user_label", "other")
            if lbl == "hardhat":
                stats["hardhat"] += 1
            elif lbl == "no_hardhat":
                stats["no_hardhat"] += 1
            elif lbl == "not_person_pump":
                stats["not_person_pump"] += 1
            elif lbl == "not_person_pipe":
                stats["not_person_pipe"] += 1
            elif lbl == "not_person_stand":
                stats["not_person_stand"] += 1
            elif "not_person" in lbl:
                stats["not_person_other"] += 1
            elif lbl == "multiple_people":
                stats["multiple_people"] += 1

        stats["total_hard_negatives"] = (
            stats["not_person_pump"] + stats["not_person_pipe"] + 
            stats["not_person_stand"] + stats["not_person_other"]
        )
        return stats

    def export_yolo_dataset(self) -> Dict:
        """
        Exports all active learning samples as a YOLO dataset directory
        with standard normalized labels for fine-tuning.
        """
        manifest = self.get_manifest()
        export_dir = os.path.join(STORE_DIR, "yolo_export")
        images_dir = os.path.join(export_dir, "images")
        labels_dir = os.path.join(export_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # Classes: 0: hardhat, 1: no_hardhat
        exported_count = 0
        for s in manifest:
            if s.get("user_label") not in ("hardhat", "no_hardhat"):
                continue

            frame_file = os.path.join(STORE_DIR, s.get("frame_url", "").replace("/learning_store/", "").replace("/", os.sep))
            if not os.path.exists(frame_file):
                continue

            frame_img = cv2.imread(frame_file)
            if frame_img is None:
                continue

            h, w = frame_img.shape[:2]
            x1, y1, x2, y2 = s["bbox"]
            xc = ((x1 + x2) / 2.0) / w
            yc = ((y1 + y2) / 2.0) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            cls_id = 0 if s["user_label"] == "hardhat" else 1

            sample_id = s["id"]
            out_img = os.path.join(images_dir, f"{sample_id}.jpg")
            out_txt = os.path.join(labels_dir, f"{sample_id}.txt")

            cv2.imwrite(out_img, frame_img)
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

            exported_count += 1

        # Generate dataset.yaml for Ultralytics YOLO fine-tuning
        yaml_file = os.path.join(export_dir, "dataset.yaml")
        yaml_content = f"""path: {export_dir.replace(os.sep, '/')}
train: images
val: images

names:
  0: hardhat
  1: no_hardhat
"""
        with open(yaml_file, "w", encoding="utf-8") as f:
            f.write(yaml_content)

        train_cmd = f"yolo detect train data={yaml_file.replace(os.sep, '/')} model=cv_core/weights/ppe_detector_best.pt epochs=25 imgsz=640 batch=8"

        return {
            "exported_samples": exported_count,
            "export_path": export_dir,
            "yaml_file": yaml_file,
            "train_command": train_cmd
        }

    def tag_dataset_crop(self, filename: str, new_label: str) -> Dict:
        """
        Updates crop annotation in dataset/verified_labels.json and/or manifest.json.
        new_label can be: 'hardhat', 'no_hardhat', 'safety_vest', 'not_person', 'multiple_people', etc.
        """
        res = self.tag_batch([filename], new_label)
        res["filename"] = os.path.basename(filename)
        return res

    def tag_batch(self, items: List[Union[str, Dict]], new_label: str) -> Dict:
        """
        Updates crop annotations in batch for:
        1) dataset/verified_labels.json (MobileNet baseline crop bank)
        2) backend/data/active_learning_store/manifest.json (Golden Active Learning store)
        """
        target_filenames = set()
        target_ids = set()

        for it in items:
            if isinstance(it, dict):
                if it.get("filename"):
                    target_filenames.add(os.path.basename(it["filename"]))
                if it.get("id"):
                    target_ids.add(str(it["id"]))
            elif isinstance(it, str):
                if it.startswith("al-"):
                    target_ids.add(it)
                else:
                    target_filenames.add(os.path.basename(it))

        category_map = {
            "hardhat": "В каске (Рабочий)",
            "no_hardhat": "Без каски (Нарушитель)",
            "not_person_pump": "⚙️ Оборудование (Насосы / Приводы)",
            "not_person_pipe": "🔵 Трубопроводы (Трубы / Задвижки)",
            "not_person_stand": "🏗️ Стенд / Металлоконструкция",
            "not_person": "🚫 Фоновый артефакт (Не человек)",
            "multiple_people": "👥 Склейка (Группа)",
            "safety_vest": "🦺 Сигнальный жилет",
            "no_safety_vest": "👕 Без жилета / Спецодежда"
        }

        # 1. Update manifest.json if any matching samples
        al_updated_count = 0
        manifest = self.get_manifest()
        manifest_changed = False
        for s in manifest:
            s_crop_name = os.path.basename(s.get("crop_url", ""))
            if s.get("id") in target_ids or s_crop_name in target_filenames:
                s["user_label"] = new_label
                s["category_title"] = category_map.get(new_label, new_label)
                s["is_person"] = new_label in ("hardhat", "no_hardhat", "multiple_people")
                manifest_changed = True
                al_updated_count += 1

        if manifest_changed:
            self._save_manifest(manifest)

        # 2. Update verified_labels.json if any matching baseline crops
        baseline_updated_count = 0
        verified_path = os.path.join(BASE_DIR, "dataset", "verified_labels.json")
        if os.path.exists(verified_path):
            try:
                with open(verified_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                is_hardhat = (new_label == "hardhat")
                is_person = (new_label not in ("not_person", "not_person_pump", "not_person_pipe", "not_person_stand"))
                is_single = (new_label != "multiple_people")
                has_vest = (new_label == "safety_vest")

                found_filenames = set()
                for it in data.get("items", []):
                    fname = os.path.basename(it.get("filename", ""))
                    if fname in target_filenames:
                        found_filenames.add(fname)
                        it["status"] = new_label
                        it["hardhat"] = is_hardhat
                        it["is_person"] = is_person
                        it["is_single"] = is_single
                        if has_vest:
                            it["safety_vest"] = True
                        it["was_modified"] = True
                        baseline_updated_count += 1

                for fname in target_filenames:
                    if fname not in found_filenames and any(fname.lower().endswith(ext) for ext in ('.jpg', '.jpeg', '.png')):
                        data.setdefault("items", []).append({
                            "filename": fname,
                            "status": new_label,
                            "hardhat": is_hardhat,
                            "is_person": is_person,
                            "is_single": is_single,
                            "safety_vest": has_vest,
                            "was_modified": True
                        })
                        baseline_updated_count += 1

                # Recalculate summary stats
                h_cnt = sum(1 for x in data["items"] if x.get("hardhat") is True and x.get("is_person") is True)
                no_h_cnt = sum(1 for x in data["items"] if x.get("hardhat") is False and x.get("is_person") is True)
                np_cnt = sum(1 for x in data["items"] if x.get("is_person") is False)
                crowd_cnt = sum(1 for x in data["items"] if x.get("is_single") is False)
                mod_cnt = sum(1 for x in data["items"] if x.get("was_modified") is True)

                data["total"] = len(data["items"])
                data["hardhat_count"] = h_cnt
                data["no_hardhat_count"] = no_h_cnt
                data["not_person_count"] = np_cnt
                data["multiple_people_count"] = crowd_cnt
                data["modified_count"] = mod_cnt

                with open(verified_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Error updating verified_labels.json: {e}")

        return {
            "status": "ok",
            "new_label": new_label,
            "updated_al_count": al_updated_count,
            "updated_baseline_count": baseline_updated_count,
            "total_updated": al_updated_count + baseline_updated_count
        }

al_manager = ActiveLearningManager()

