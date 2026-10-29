import os
import sys
import glob
import time
import json
import threading
import cv2
import numpy as np
from PIL import Image
import re
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from cv_core.pipeline import RVCSafetyPipeline
from backend.storage import storage
from backend.active_learning import al_manager
from backend.contractor_detector import contractor_detector

CORRECTIONS_FILE = os.path.join(BASE_DIR, "backend", "data", "active_learning_corrections.json")

class LiveStreamer:
    def __init__(self):
        self.pipeline = RVCSafetyPipeline(zones=[])
        self.playlists = {}
        self.current_playlist = "golden"
        self.frame_paths = []
        self._load_playlists()
        
        self.current_frame_idx = 0
        self.is_running = False
        self.is_paused = False
        self.latest_annotated_jpeg = None
        self.latest_raw_jpeg = None
        self.current_bgr_frame = None
        self.current_detections = []
        self.current_img_path = ""
        self.lock = threading.RLock()
        
        # Telemetry
        self.fps = 2.0
        self.latency_ms = 25.0
        self.active_tracks_count = 0
        self.total_incidents_count = 0

        # Real-time session overrides (suppressions & manual boxes)
        self.suppressed_tracks = set()
        self.suppressed_bboxes_by_frame = {}
        self.manual_detections = {}
        self.user_label_overrides_by_frame = {}
        self.global_suppressed_artifacts = []
        self.show_manual_overlays = False
        self.auto_loop = False  # False = pause on the last frame of the playlist for clean review

        # Restore previous user corrections from Active Learning store
        self.reload_corrections(reprocess=False)
        
        # Start background streamer thread
        self.thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.thread.start()

    @staticmethod
    def _calc_iou(boxA, boxB) -> float:
        if not boxA or not boxB or len(boxA) < 4 or len(boxB) < 4:
            return 0.0
        try:
            bA = [float(v) for v in boxA]
            bB = [float(v) for v in boxB]
            xA = max(bA[0], bB[0])
            yA = max(bA[1], bB[1])
            xB = min(bA[2], bB[2])
            yB = min(bA[3], bB[3])
            interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
            if interArea <= 0.0:
                return 0.0
            boxAArea = max(1e-5, (bA[2] - bA[0]) * (bA[3] - bA[1]))
            boxBArea = max(1e-5, (bB[2] - bB[0]) * (bB[3] - bB[1]))
            return float(interArea / float(boxAArea + boxBArea - interArea))
        except Exception:
            return 0.0

    def _get_next_manual_id(self) -> str:
        max_num = 0
        with self.lock:
            for f_dets in self.manual_detections.values():
                for d in f_dets:
                    tid = str(d.get("track_id", ""))
                    m = re.search(r"[МM]-?(\d+)", tid)
                    if m:
                        try:
                            max_num = max(max_num, int(m.group(1)))
                        except ValueError:
                            pass
        try:
            for s in al_manager.get_all_samples():
                tid = str(s.get("track_id", ""))
                m = re.search(r"[МM]-?(\d+)", tid)
                if m:
                    try:
                        max_num = max(max_num, int(m.group(1)))
                    except ValueError:
                        pass
        except Exception:
            pass
        return f"М-{max_num + 1}"

    def reload_corrections(self, reprocess: bool = True):
        with self.lock:
            self.suppressed_tracks.clear()
            self.suppressed_bboxes_by_frame.clear()
            self.manual_detections.clear()
            self.user_label_overrides_by_frame.clear()
            self.global_suppressed_artifacts.clear()
            # Keep session overlays clean on startup so operator tests pure Model V2 inference.
            # Historical Active Learning samples belong in the training pool (al_manager),
            # not as artificial live overrides on top of the trained neural network.
            self.pipeline.suppressed_tracks = self.suppressed_tracks
            self.pipeline.suppressed_bboxes_by_frame = self.suppressed_bboxes_by_frame
            self.pipeline.user_label_overrides_by_frame = self.user_label_overrides_by_frame
            self.pipeline.global_suppressed_artifacts = self.global_suppressed_artifacts

            if reprocess and self.frame_paths and 0 <= self.current_frame_idx < len(self.frame_paths):
                self._process_single_frame(self.current_frame_idx)

    def _load_playlists(self):
        # 1. Golden 247 split
        manifest_path = os.path.join(BASE_DIR, "dataset", "splits_manifest.json")
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            golden = manifest["splits"]["train"] + manifest["splits"]["val"] + manifest["splits"]["test"]
            self.playlists["golden"] = [p for p in golden if os.path.exists(p)]
        except Exception as e:
            print("Error loading splits_manifest:", e)
            self.playlists["golden"] = []

        # 2. All 5066 raw frames
        all_raw = []
        for pat in ["rvk-ai-vision-all-raw-*/raw/*.jpg"]:
            all_raw.extend(glob.glob(os.path.join(BASE_DIR, pat)))
        all_raw = sorted(all_raw)
        self.playlists["all_5000"] = all_raw

        # 3. Specific shifts
        self.playlists["2026__08__21"] = [p for p in all_raw if "2026__08__21" in p]
        self.playlists["2026__08__24"] = [p for p in all_raw if "2026__08__24" in p]
        self.playlists["2026__08__29"] = [p for p in all_raw if "2026__08__29" in p]

        # 4. Smoke & Fire test playlist
        sf_frames = glob.glob(os.path.join(BASE_DIR, "WATER_SMOKE_FIRE_GENERATOR_V01", "output", "dataset_v01", "yolo_bbox", "images", "*", "*.jpg"))
        if sf_frames:
            self.playlists["smoke_fire_test"] = sorted(sf_frames)

        # 5. Vests test playlist
        vests_frames = glob.glob(os.path.join(BASE_DIR, "dataset", "test_playlists", "vests", "*.jpg"))
        if vests_frames:
            self.playlists["vests_test"] = sorted(vests_frames)

        # 6. Smoking test playlist
        smoking_frames = glob.glob(os.path.join(BASE_DIR, "dataset", "test_playlists", "smoking", "*.jpg"))
        if smoking_frames:
            self.playlists["smoking_test"] = sorted(smoking_frames)

        # 7. Phone in danger zone test playlist
        phone_frames = glob.glob(os.path.join(BASE_DIR, "dataset", "test_playlists", "phone", "*.jpg"))
        if phone_frames:
            self.playlists["phone_test"] = sorted(phone_frames)

        # 8. Real Industrial Cameras Streams (4 new cameras from videos folder)
        for cam_key in ["cam_01", "cam_03", "cam_04", "cam_05"]:
            cam_dir = os.path.join(BASE_DIR, "dataset", "camera_frames", cam_key)
            cam_frames = sorted(glob.glob(os.path.join(cam_dir, "*.jpg")))
            if cam_frames:
                self.playlists[cam_key] = cam_frames

        self.frame_paths = self.playlists.get(self.current_playlist, self.playlists["golden"])
        print(f"Playlists loaded: Golden({len(self.playlists.get('golden', []))}), "
              f"Vests({len(self.playlists.get('vests_test', []))}), "
              f"Smoking({len(self.playlists.get('smoking_test', []))}), "
              f"Phone({len(self.playlists.get('phone_test', []))}), "
              f"SmokeFire({len(self.playlists.get('smoke_fire_test', []))}), "
              f"All 5000({len(all_raw)}), Shift 21.08({len(self.playlists.get('2026__08__21', []))})")

    def set_playlist(self, name: str):
        with self.lock:
            if name in self.playlists and self.playlists[name]:
                self.current_playlist = name
                self.frame_paths = self.playlists[name]
                self.current_frame_idx = 0
                if self.frame_paths:
                    self._process_single_frame(0)
                return True
        return False

    def set_speed(self, multiplier: float):
        with self.lock:
            multiplier = max(0.25, min(10.0, float(multiplier)))
            self.fps = 2.0 * multiplier
            return self.fps

    def set_auto_loop(self, enabled: bool):
        with self.lock:
            self.auto_loop = bool(enabled)
            return self.auto_loop

    def toggle_pause(self, paused=None):
        with self.lock:
            if paused is None:
                self.is_paused = not self.is_paused
            else:
                self.is_paused = bool(paused)
            # If resuming playback while already on the last frame, start over from 0
            if not self.is_paused and self.frame_paths and self.current_frame_idx >= len(self.frame_paths) - 1:
                self.current_frame_idx = 0
            if self.is_paused and self.frame_paths:
                self._process_single_frame(self.current_frame_idx)
            return self.is_paused

    def step_frame(self, delta: int = 1):
        with self.lock:
            if self.frame_paths:
                self.current_frame_idx = (self.current_frame_idx + delta) % len(self.frame_paths)
                self._process_single_frame(self.current_frame_idx)
            return self.current_frame_idx

    def seek_frame(self, idx: int):
        with self.lock:
            if self.frame_paths:
                self.current_frame_idx = max(0, min(len(self.frame_paths) - 1, int(idx)))
                self.is_paused = True
                self._process_single_frame(self.current_frame_idx)
            return self.current_frame_idx

    def _process_single_frame(self, idx: int):
        img_path = self.frame_paths[idx]
        frame_name = os.path.basename(img_path)
        bgr = cv2.imread(img_path)
        if bgr is None:
            return

        active_zones = [z for z in storage.get_zones() if z.get("is_active", True)]
        if not getattr(self.pipeline, "detect_danger_zones", True):
            active_zones = []
        self.pipeline.zones = active_zones
        self.pipeline.suppressed_tracks = self.suppressed_tracks

        # Inform pipeline of manual boxes on this frame so it does not draw duplicate auto-detections (only when manual overlays are active)
        frame_suppressed = list(self.suppressed_bboxes_by_frame.get(frame_name, []))
        if getattr(self, "show_manual_overlays", False):
            for m in self.manual_detections.get(frame_name, []):
                frame_suppressed.append(m["bbox"])
        self.pipeline.suppressed_bboxes_by_frame[frame_name] = frame_suppressed
        self.pipeline.user_label_overrides_by_frame = self.user_label_overrides_by_frame

        _, raw_buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        t0 = time.time()
        ann_pil, incidents = self.pipeline.process_frame(
            pil_img, frame_idx=idx, timestamp=time.time(), frame_name=frame_name
        )
        try:
            self.current_contractor_detections = contractor_detector.detect(bgr, conf_thresh=0.22)
        except Exception:
            self.current_contractor_detections = []
        self.latency_ms = round((time.time() - t0) * 1000, 1)

        if incidents:
            camera_map = {
                "golden": "ЦМОО • Шнековые сепараторы (Golden)",
                "smoke_fire_test": "Цех насосов №2 • Термоконтроль (Задымление/Огонь)",
                "vests_test": "Цех решеток №1 • Контроль сигнальных жилетов",
                "smoking_test": "Участок обезвоживания • Контроль курения",
                "phone_test": "Шнековый транспортер • Смартфон в опасной зоне",
                "2026__08__21": "Осадкоуплотнитель №1 (21 авг)",
                "2026__08__24": "Машинный зал насосов (24 авг)",
                "2026__08__29": "Транспортер кека (29 авг)",
                "all_5000": "Боксы АВМ (5000 кадров)"
            }
            cam_title = camera_map.get(self.current_playlist, "ЦМОО • Сепараторы")
            model_ver = getattr(self.pipeline, "active_model_name", "v2_cascade")
            for inc in incidents:
                storage.add_incident({
                    "timestamp": inc["timestamp"],
                    "time_str": datetime.fromtimestamp(inc["timestamp"]).strftime("%H:%M:%S"),
                    "frame_idx": inc["frame_idx"],
                    "track_id": inc["track_id"],
                    "violations": inc["violations"],
                    "bbox": inc["bbox"],
                    "playlist": self.current_playlist,
                    "camera_title": cam_title,
                    "model_version": model_ver
                })
                self.total_incidents_count += 1

        ann_bgr = cv2.cvtColor(np.array(ann_pil), cv2.COLOR_RGB2BGR)

        # 1. Pipeline auto-detections for this frame
        auto_dets = list(getattr(self.pipeline, "last_frame_detections", []))

        # 2. Add manual detections for THIS SPECIFIC frame only (excluding any suppressed equipment artifacts)
        valid_manual = []
        if getattr(self, "show_manual_overlays", False):
            manual_list = self.manual_detections.get(frame_name, [])
            for m in manual_list:
                mbox = m.get("bbox", [])
                is_supp = False
                for sbox in self.global_suppressed_artifacts:
                    if self._calc_iou(mbox, sbox) > 0.4:
                        is_supp = True
                        break
                if not is_supp:
                    valid_manual.append(m)

        # 3. Deduplicate: if an auto-detection overlaps with a manual box (IoU > 0.35), the manual box takes precedence!
        filtered_dets = []
        for ad in auto_dets:
            abox = ad.get("bbox", [])
            overlaps = any(self._calc_iou(abox, vm.get("bbox", [])) > 0.35 for vm in valid_manual)
            if not overlaps:
                filtered_dets.append(ad)

        # Append authoritative manual detections
        filtered_dets.extend(valid_manual)

        # 3. Visually draw manual worker boxes for this specific frame only
        for m in valid_manual:
            mx1, my1, mx2, my2 = [int(v) for v in m["bbox"]]
            m_col = (34, 197, 94) if m.get("has_hardhat", False) else (239, 68, 68)
            cv2.rectangle(ann_bgr, (mx1, my1), (mx2, my2), m_col, 2)
            clean_tid = str(m.get("track_id", "")).replace("manual_", "М-")
            badge_t = f"ID:{clean_tid} | " + ("В каске (Ручн)" if m.get("has_hardhat") else "Без каски (Ручн)")
            tb_top = max(0, my1 - 24)
            cv2.rectangle(ann_bgr, (mx1, tb_top), (mx1 + 220, my1), (15, 23, 42), -1)
            cv2.rectangle(ann_bgr, (mx1, tb_top), (mx1 + 220, my1), m_col, 2)
            cv2.putText(ann_bgr, badge_t, (mx1 + 6, my1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # HUD banner
        hud_text = f"CAM-01 | {self.current_playlist.upper()} | Frame: {idx+1}/{len(self.frame_paths)} | Speed: {self.fps/2:.1f}x"
        cv2.rectangle(ann_bgr, (20, 20), (740, 65), (15, 23, 42), -1)
        cv2.rectangle(ann_bgr, (20, 20), (740, 65), (59, 130, 246), 2)
        cv2.putText(ann_bgr, hud_text, (35, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)

        _, ann_buf = cv2.imencode(".jpg", ann_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])

        self.latest_raw_jpeg = raw_buf.tobytes()
        self.latest_annotated_jpeg = ann_buf.tobytes()
        self.current_bgr_frame = bgr.copy()
        self.current_detections = filtered_dets
        self.current_img_path = img_path
        self.active_tracks_count = len(filtered_dets)

    def _stream_loop(self):
        self.is_running = True

        while self.is_running:
            if not self.frame_paths or self.is_paused:
                time.sleep(0.05)
                continue

            loop_start = time.time()
            try:
                with self.lock:
                    idx = self.current_frame_idx
                    if idx + 1 >= len(self.frame_paths):
                        self._process_single_frame(idx)
                        if getattr(self, "auto_loop", False):
                            self.current_frame_idx = 0
                        else:
                            self.is_paused = True
                            self.current_frame_idx = len(self.frame_paths) - 1
                            print(f"[Streamer] Reached end of playlist '{self.current_playlist}' ({len(self.frame_paths)} frames). Paused.")
                    else:
                        self.current_frame_idx = idx + 1
                        self._process_single_frame(idx)
            except Exception as e:
                print("Stream loop frame processing error:", e)

            interval = 1.0 / max(0.2, self.fps)
            elapsed = time.time() - loop_start
            to_sleep = max(0.005, interval - elapsed)
            time.sleep(to_sleep)

    def get_annotated_frame(self):
        with self.lock:
            return self.latest_annotated_jpeg

    def get_raw_frame(self):
        with self.lock:
            return self.latest_raw_jpeg

    def get_frame_state(self):
        with self.lock:
            cur_weights = getattr(self.pipeline, "current_weights_path", "")
            is_v2 = "active_run" in cur_weights or "v2" in cur_weights or "training_runs" in cur_weights
            model_tag = "v2 (Active Learning)" if is_v2 else "v1 (Baseline)"

            return {
                "frame_idx": self.current_frame_idx,
                "total_frames": len(self.frame_paths),
                "playlist": self.current_playlist,
                "is_paused": self.is_paused,
                "auto_loop": getattr(self, "auto_loop", False),
                "latency_ms": self.latency_ms,
                "active_tracks": self.active_tracks_count,
                "detections": self.current_detections,
                "contractor_detections": getattr(self, "current_contractor_detections", []),
                "img_filename": os.path.basename(self.current_img_path),
                "show_manual_overlays": getattr(self, "show_manual_overlays", False),
                "active_classifier": getattr(self.pipeline, "active_classifier_name", "efficientnet_b0"),
                "active_model_tag": model_tag
            }

    def toggle_manual_overlays(self, enabled: bool = None) -> bool:
        with self.lock:
            if enabled is None:
                self.show_manual_overlays = not getattr(self, "show_manual_overlays", False)
            else:
                self.show_manual_overlays = bool(enabled)
            self._process_single_frame(self.current_frame_idx)
            return self.show_manual_overlays

    def clear_manual_overrides(self):
        with self.lock:
            self.manual_detections.clear()
            self.user_label_overrides_by_frame.clear()
            self._process_single_frame(self.current_frame_idx)
            return {"status": "ok", "message": "Сессионные ручные боксы очищены"}

    def save_correction(self, data: dict):
        with self.lock:
            frame_bgr = None
            if self.current_bgr_frame is not None:
                frame_bgr = self.current_bgr_frame.copy()
            elif self.current_img_path and os.path.exists(self.current_img_path):
                frame_bgr = cv2.imread(self.current_img_path)

            if frame_bgr is None:
                frame_bgr = np.zeros((1080, 1920, 3), dtype=np.uint8)

            bbox = data.get("bbox", [0, 0, 100, 100])
            user_label = data.get("label", "not_person")
            track_id = data.get("track_id", "manual")
            model_prediction = data.get("model_prediction", None)

            if not model_prediction and self.current_detections:
                for d in self.current_detections:
                    if str(d.get("track_id")) == str(track_id):
                        model_prediction = {
                            "has_hardhat": d.get("has_hardhat", False),
                            "confidence": d.get("hh_conf", 0.0)
                        }
                        break

            source_frame = os.path.basename(self.current_img_path) if self.current_img_path else "camera_frame.jpg"

            sample_record = al_manager.save_sample(
                frame_bgr=frame_bgr,
                source_frame_name=source_frame,
                playlist=self.current_playlist,
                bbox=bbox,
                user_label=user_label,
                track_id=track_id,
                model_prediction=model_prediction
            )

            # Also maintain legacy active_learning_corrections.json for compatibility
            try:
                legacy_list = []
                if os.path.exists(CORRECTIONS_FILE):
                    with open(CORRECTIONS_FILE, "r", encoding="utf-8") as f:
                        legacy_list = json.load(f)
                legacy_list.append(sample_record)
                with open(CORRECTIONS_FILE, "w", encoding="utf-8") as f:
                    json.dump(legacy_list, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

            # Update live session overrides
            if user_label.startswith("not_person"):
                tid_str = str(track_id) if track_id is not None else ""
                if tid_str and not tid_str.startswith("manual") and not tid_str.startswith("М-") and tid_str not in ("new", "manual"):
                    self.suppressed_tracks.add(tid_str)
                self.suppressed_bboxes_by_frame.setdefault(source_frame, []).append([float(v) for v in bbox])
                # Add to global static suppressed equipment artifacts (camera-wide across all frames)
                self.global_suppressed_artifacts.append([float(v) for v in bbox])
                self.pipeline.global_suppressed_artifacts = self.global_suppressed_artifacts

                # Immediately purge all manual detections overlapping this bbox or matching tid_str across all frames!
                for fname in list(self.manual_detections.keys()):
                    self.manual_detections[fname] = [
                        d for d in self.manual_detections[fname]
                        if str(d.get("track_id")) != tid_str and self._calc_iou(d.get("bbox", []), bbox) < 0.4
                    ]
                    if not self.manual_detections[fname]:
                        del self.manual_detections[fname]

                # Also remove from user_label_overrides_by_frame
                for fname in list(self.user_label_overrides_by_frame.keys()):
                    self.user_label_overrides_by_frame[fname] = [
                        o for o in self.user_label_overrides_by_frame[fname]
                        if str(o.get("track_id")) != tid_str and self._calc_iou(o.get("bbox", []), bbox) < 0.4
                    ]

            elif user_label in ("hardhat", "no_hardhat"):
                self.user_label_overrides_by_frame.setdefault(source_frame, []).append({
                    "track_id": track_id,
                    "bbox": [float(v) for v in bbox],
                    "has_hardhat": (user_label == "hardhat")
                })
                self.pipeline.user_label_overrides_by_frame = self.user_label_overrides_by_frame
                if str(track_id).startswith("manual") or str(track_id).startswith("M") or str(track_id).startswith("М") or str(track_id) == "new":
                    existing_list = self.manual_detections.setdefault(source_frame, [])
                    # Check if updating an existing manual detection on this frame (by ID or high IoU > 0.6)
                    target_idx = -1
                    for i, ed in enumerate(existing_list):
                        if str(ed.get("track_id")) == str(track_id) or self._calc_iou(ed.get("bbox", []), bbox) > 0.6:
                            target_idx = i
                            break

                    if target_idx >= 0:
                        manual_id = existing_list[target_idx].get("track_id", track_id)
                        existing_list[target_idx] = {
                            "track_id": manual_id,
                            "bbox": [float(v) for v in bbox],
                            "has_hardhat": (user_label == "hardhat"),
                            "hh_conf": 1.0,
                            "zone_violation": False,
                            "is_manual": True
                        }
                    else:
                        manual_id = track_id if (str(track_id).startswith("М-") or str(track_id).startswith("M-")) else self._get_next_manual_id()
                        manual_det = {
                            "track_id": manual_id,
                            "bbox": [float(v) for v in bbox],
                            "has_hardhat": (user_label == "hardhat"),
                            "hh_conf": 1.0,
                            "zone_violation": False,
                            "is_manual": True
                        }
                        existing_list.append(manual_det)

            # Immediately re-process the frame so snapshot and latest_annotated_jpeg update at once
            if self.frame_paths and 0 <= self.current_frame_idx < len(self.frame_paths):
                self._process_single_frame(self.current_frame_idx)

            return sample_record

    def delete_detection_box(self, track_id: str, bbox: list = None, source_frame: str = None):
        with self.lock:
            tid_str = str(track_id) if track_id is not None else ""
            deleted_count = 0

            # 1. Purge from manual_detections across all frames
            for fname in list(self.manual_detections.keys()):
                orig_len = len(self.manual_detections[fname])
                self.manual_detections[fname] = [
                    d for d in self.manual_detections[fname]
                    if str(d.get("track_id")) != tid_str and not (
                        bbox and len(bbox) == 4 and self._calc_iou(d.get("bbox", []), bbox) > 0.4
                    )
                ]
                deleted_count += (orig_len - len(self.manual_detections[fname]))
                if not self.manual_detections[fname]:
                    del self.manual_detections[fname]

            # 2. If it was an auto track (e.g. ID: 5), suppress it so detector won't re-spawn it on equipment
            if tid_str and not tid_str.startswith("manual") and not tid_str.startswith("М-") and tid_str not in ("new", "manual"):
                self.suppressed_tracks.add(tid_str)
                if source_frame and bbox and len(bbox) == 4:
                    self.suppressed_bboxes_by_frame.setdefault(source_frame, []).append([float(v) for v in bbox])
                if bbox and len(bbox) == 4:
                    self.global_suppressed_artifacts.append([float(v) for v in bbox])
                    self.pipeline.global_suppressed_artifacts = self.global_suppressed_artifacts

            # 3. Purge from user_label_overrides_by_frame
            for fname in list(self.user_label_overrides_by_frame.keys()):
                self.user_label_overrides_by_frame[fname] = [
                    o for o in self.user_label_overrides_by_frame[fname]
                    if str(o.get("track_id")) != tid_str and not (
                        bbox and len(bbox) == 4 and self._calc_iou(o.get("bbox", []), bbox) > 0.4
                    )
                ]

            # 4. Delete corresponding sample from active learning store
            try:
                for s in al_manager.get_manifest():
                    if str(s.get("track_id")) == tid_str or (
                        bbox and len(bbox) == 4 and self._calc_iou(s.get("bbox", []), bbox) > 0.4
                    ):
                        al_manager.delete_sample(s.get("id"))
            except Exception as e:
                print("Error purging sample on delete_box:", e)

            # 5. Re-process current frame immediately
            if self.frame_paths and 0 <= self.current_frame_idx < len(self.frame_paths):
                self._process_single_frame(self.current_frame_idx)

            return {"status": "ok", "deleted_track": tid_str, "purged": deleted_count}

    def undo_last_correction(self):
        with self.lock:
            samples = al_manager.get_manifest()
            if not samples:
                return None
            last_sample = samples[0]
            sample_id = last_sample.get("id")
            if not sample_id:
                return None
            al_manager.delete_sample(sample_id)
            self.reload_corrections(reprocess=True)
            return last_sample

streamer = LiveStreamer()
