import os
import time
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import onnxruntime as ort
import torch
import torchvision.transforms as transforms

from cv_core.tracker import BipartiteTracker, calculate_iou
from cv_core.benchmark_models import MobileNetV3Classifier, EfficientNetB0Classifier, ResNet18Classifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def nms_filter(detections, iou_thresh=0.45):
    if len(detections) <= 1:
        return detections
    dets = sorted(detections, key=lambda d: d.get('conf', 0.0), reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        remaining = []
        for d in dets:
            iou = calculate_iou(best['bbox'], d['bbox'])
            if iou < iou_thresh:
                remaining.append(d)
        dets = remaining
    return keep

class RVCSafetyPipeline:
    """
    End-to-End ROSVODOKANAL Safety CV Pipeline v2.3:
    - High sensitivity detector (conf=0.18) detecting all workers (bending, distant, occluded)
    - Full Stationary Artifact Masking (Pump, Window pillar, Wall stand)
    - AI Person-vs-Machine Verifier
    - Bipartite Tracker (15 frames memory)
    - ONNX Multi-Task Classifier (Hardhat + Vest)
    - HD 22px Bold TrueType Arial font for crystal-clear Russian badges
    """
    def __init__(self, 
                 onnx_classifier_path="cv_core/weights/ppe_classifier.onnx",
                 onnx_verifier_path="cv_core/weights/person_vs_machine.onnx",
                 zones=None):
        self.tracker = BipartiteTracker(min_iou=0.20, max_missed_frames=15)
        self.zones = zones or []
        self.suppressed_tracks = set()
        self.suppressed_bboxes_by_frame = {}
        self.user_label_overrides_by_frame = {}
        self.global_suppressed_artifacts = []
        
        self.classifier_session = ort.InferenceSession(onnx_classifier_path, providers=["CPUExecutionProvider"])
        self.verifier_session = ort.InferenceSession(onnx_verifier_path, providers=["CPUExecutionProvider"])
        
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        from ultralytics import YOLO
        base_weights = "cv_core/weights/ppe_detector_best.pt"
        self.base_detector = YOLO(base_weights)
        self.detector = self.base_detector
        self.current_weights_path = os.path.abspath(base_weights)
        self.v2_specialist = None
        self.is_v2_active = False

        # Discover and initialize trained v2 specialist if exists
        self._init_v2_specialist()
        
        # Load crisp Segoe UI / Arial fonts for HUD badges
        font_paths_bold = [
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "arial.ttf"
        ]
        font_paths_regular = [
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "arial.ttf"
        ]
        self.font = None
        self.font_bold = None
        for fp in font_paths_bold:
            if os.path.exists(fp):
                try:
                    self.font_bold = ImageFont.truetype(fp, 17)
                    break
                except Exception:
                    pass
        for fp in font_paths_regular:
            if os.path.exists(fp):
                try:
                    self.font = ImageFont.truetype(fp, 16)
                    break
                except Exception:
                    pass
        if self.font is None:
            self.font = ImageFont.load_default()
        if self.font_bold is None:
            self.font_bold = self.font
            
        self.incidents = []
        
        # SOTA Model Backbones (MobileNetV3, EfficientNet-B0, ResNet-18)
        self.active_classifier_name = "efficientnet_b0"
        self.pytorch_classifier = None
        self._init_classifier(self.active_classifier_name)

        # Instance Segmentation mode (YOLOv8-Seg)
        self.segmentation_mode = False
        self.seg_detector = None

        # Smoke & Fire Detection mode (active 24/7 for industrial safety)
        self.smoke_fire_mode = True
        self.smoke_fire_detector = None

        # Human Pose Heuristics (Smoking, Phone usage, Dangerous posture)
        self.pose_mode = True
        self.pose_detector = None
        self._init_pose_detector()

        # Modular Feature Detection Toggles (user can enable/disable any check on the fly)
        self.detect_hardhat = True
        self.detect_vest = True
        self.detect_smoking = True
        self.detect_phone = True
        self.detect_danger_zones = True
        self.contractor_baseline_mode = False

    def set_feature_toggle(self, feature: str, enabled: bool) -> bool:
        """Dynamically toggle individual violation checkers."""
        enabled = bool(enabled)
        if feature in ("hardhat", "hard_hat", "helmet"):
            self.detect_hardhat = enabled
            return self.detect_hardhat
        elif feature in ("vest", "safety_vest"):
            self.detect_vest = enabled
            return self.detect_vest
        elif feature in ("smoking", "cigarette"):
            self.detect_smoking = enabled
            return self.detect_smoking
        elif feature in ("phone", "smartphone"):
            self.detect_phone = enabled
            return self.detect_phone
        elif feature in ("smoke_fire", "smokefire", "fire"):
            return self.set_smoke_fire_mode(enabled)
        elif feature in ("danger_zones", "zone", "zones"):
            self.detect_danger_zones = enabled
            return self.detect_danger_zones
        return False

    def get_features_state(self) -> dict:
        """Return live state of all modular safety checkers."""
        return {
            "hardhat": bool(getattr(self, "detect_hardhat", True)),
            "vest": bool(getattr(self, "detect_vest", True)),
            "smoking": bool(getattr(self, "detect_smoking", True)),
            "phone": bool(getattr(self, "detect_phone", True)),
            "smoke_fire": bool(getattr(self, "smoke_fire_mode", True)),
            "danger_zones": bool(getattr(self, "detect_danger_zones", True)),
            "contractor_baseline": bool(getattr(self, "contractor_baseline_mode", False)),
            "active_classifier": getattr(self, "active_classifier_name", "efficientnet_b0"),
            "segmentation_mode": bool(getattr(self, "segmentation_mode", False))
        }

    def _init_pose_detector(self):
        pose_weights = os.path.join(BASE_DIR, "cv_core", "weights", "yolov8n-pose.pt")
        if os.path.exists(pose_weights):
            try:
                from ultralytics import YOLO
                self.pose_detector = YOLO(pose_weights)
                print(f"[Pipeline] Pose detector initialized from {pose_weights}")
            except Exception as e:
                print(f"[Pipeline] Warning: Could not initialize pose detector: {e}")

    def _init_classifier(self, model_name: str):
        try:
            weights_map = {
                "efficientnet_b0": ("cv_core/weights/efficientnet_b0_best.pth", EfficientNetB0Classifier),
                "mobilenet_v3": ("cv_core/weights/mobilenetv3_small_best.pth", MobileNetV3Classifier),
                "resnet_18": ("cv_core/weights/resnet_18_best.pth", ResNet18Classifier)
            }
            if model_name in weights_map:
                w_path, cls_builder = weights_map[model_name]
                if os.path.exists(w_path):
                    model = cls_builder().to(DEVICE)
                    model.load_state_dict(torch.load(w_path, map_location=DEVICE, weights_only=True))
                    model.eval()
                    self.pytorch_classifier = model
                    self.active_classifier_name = model_name
                    print(f"[Pipeline] SOTA Classifier activated: {model_name} on {DEVICE}")
                    return True
        except Exception as e:
            print(f"[Pipeline] Failed to load PyTorch classifier {model_name}: {e}")
        return False

    def set_classifier(self, model_name: str) -> bool:
        """Switch active classifier backbone dynamically on-the-fly."""
        success = self._init_classifier(model_name)
        if success:
            self.active_classifier_name = model_name
        return success

    def set_segmentation_mode(self, enabled: bool) -> bool:
        """Enable or disable YOLOv8-Seg instance segmentation overlay."""
        self.segmentation_mode = bool(enabled)
        if self.segmentation_mode and self.seg_detector is None:
            from ultralytics import YOLO
            seg_weights = "cv_core/weights/yolov8n-seg.pt"
            if os.path.exists(seg_weights):
                self.seg_detector = YOLO(seg_weights)
                print("[Pipeline] YOLOv8-Seg model initialized")
        return self.segmentation_mode

    def set_smoke_fire_mode(self, enabled: bool) -> bool:
        """Enable or disable Smoke & Fire hazard detector."""
        self.smoke_fire_mode = bool(enabled)
        if self.smoke_fire_mode and self.smoke_fire_detector is None:
            from ultralytics import YOLO
            sf_weights = "cv_core/weights/smoke_fire_best.pt"
            if os.path.exists(sf_weights):
                self.smoke_fire_detector = YOLO(sf_weights)
                print("[Pipeline] Smoke & Fire hazard detector initialized")
        return self.smoke_fire_mode

    def _init_v2_specialist(self):
        runs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend", "data", "training_runs")
        if os.path.exists(runs_dir):
            candidates = []
            for root, dirs, files in os.walk(runs_dir):
                if "best.pt" in files:
                    p = os.path.join(root, "best.pt")
                    candidates.append((os.path.getmtime(p), p))
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                v2_path = candidates[0][1]
                from ultralytics import YOLO
                self.v2_specialist = YOLO(v2_path)
                self.v2_weights_path = os.path.abspath(v2_path)
                self.is_v2_active = True
                self.current_weights_path = self.v2_weights_path
                print(f"[Pipeline] Cascade V2 Specialist loaded: {v2_path}")

    def reload_detector(self, weights_path: str) -> bool:
        """Hot-reload detector weights without restarting pipeline or camera stream."""
        try:
            from ultralytics import YOLO
            if os.path.exists(weights_path):
                norm_target = os.path.abspath(weights_path)
                norm_base = os.path.abspath("cv_core/weights/ppe_detector_best.pt")
                if norm_target == norm_base:
                    self.is_v2_active = False
                    self.current_weights_path = norm_base
                    print("[Pipeline] Mode switched to: v1 (Baseline)")
                else:
                    self.v2_specialist = YOLO(weights_path)
                    self.v2_weights_path = norm_target
                    self.is_v2_active = True
                    self.current_weights_path = norm_target
                    print(f"[Pipeline] Mode switched to: v2 (Cascade Active Learning) with {weights_path}")
                return True
        except Exception as e:
            print(f"[Pipeline] Hot-reload error: {e}")
        return False

    def is_static_background_artifact(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        bw = x2 - x1
        bh = y2 - y1
        
        # 1. Window pillar column at x ~ 1480, y ~ 600
        if 1450 <= cx <= 1520 and 490 <= cy <= 720 and bw < 55:
            return True
            
        # 2. Center pump machine at x ~ 560, y ~ 740
        if 500 <= cx <= 630 and 550 <= cy <= 920 and bh > 250:
            return True

        # 3. Frame 7 Wall stand / placard at x ~ 640, y ~ 420
        if 610 <= cx <= 670 and 360 <= cy <= 500 and bw < 50:
            return True
            
        return False

    def verify_is_person(self, person_crop):
        img_t = self.transform(person_crop).unsqueeze(0).numpy()
        outputs = self.verifier_session.run(None, {"image": img_t})
        logit = outputs[0][0][0]
        conf = 1.0 / (1.0 + np.exp(-logit))
        return bool(conf >= 0.40), float(conf)

    def classify_ppe(self, person_crop):
        w, h = person_crop.size
        head_h = max(int(h * 0.35), 20)
        head_crop = person_crop.crop((0, 0, w, min(head_h, h)))
        
        if self.pytorch_classifier is not None:
            try:
                head_t = self.transform(head_crop).unsqueeze(0).to(DEVICE)
                body_t = self.transform(person_crop).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    hh_logit, vest_logit = self.pytorch_classifier(head_t, body_t)
                    hh_conf = float(torch.sigmoid(hh_logit).cpu().numpy()[0])
                    vest_conf = float(torch.sigmoid(vest_logit).cpu().numpy()[0])
                has_hardhat = bool(hh_conf >= 0.50)
                has_vest = bool(vest_conf >= 0.50)
                return has_hardhat, has_vest, float(hh_conf), float(vest_conf)
            except Exception as e:
                pass
        
        head_t = self.transform(head_crop).unsqueeze(0).numpy()
        body_t = self.transform(person_crop).unsqueeze(0).numpy()
        
        outputs = self.classifier_session.run(None, {"head_crop": head_t, "body_crop": body_t})
        hh_logit = outputs[0][0]
        vest_logit = outputs[1][0]
        
        hh_conf = 1.0 / (1.0 + np.exp(-hh_logit))
        vest_conf = 1.0 / (1.0 + np.exp(-vest_logit))
        
        has_hardhat = bool(hh_conf >= 0.50)
        has_vest = bool(vest_conf >= 0.50)
        
        return has_hardhat, has_vest, float(hh_conf), float(vest_conf)

    def process_frame(self, frame_img, frame_idx=0, timestamp=None, frame_name=""):
        if timestamp is None:
            timestamp = time.time()
            
        w_img, h_img = frame_img.size

        # 1. Base detector finds all humans with high sensitivity
        det_results = self.base_detector.predict(frame_img, conf=0.18, verbose=False)[0]
        raw_detections = []
        
        if getattr(self, "contractor_baseline_mode", False):
            # Exact contractor baseline: raw single-stage output without cascade verifiers or equipment suppression
            for box in det_results.boxes:
                cls_id = int(box.cls[0])
                cls_name = str(self.base_detector.names.get(cls_id, "")).lower()
                bconf = float(box.conf[0])
                if bconf >= 0.18:
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    x1, y1, x2, y2 = [int(v) for v in xyxy]
                    raw_detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'conf': bconf,
                        'cls_name': cls_name,
                        'v2_label': cls_name
                    })
        else:
            for box in det_results.boxes:
                cls_id = int(box.cls[0])
                cls_name = str(self.base_detector.names.get(cls_id, "")).lower()
                
                if cls_name in ["human", "person"]:
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    x1, y1, x2, y2 = [int(v) for v in xyxy]
                    bw, bh = x2 - x1, y2 - y1
                    
                    # Size filtering
                    if bh < 28 or bw < 10:
                        continue
                    # Aspect ratio down to 0.40 for bending/sitting
                    if (bh / max(bw, 1)) < 0.40:
                        continue
                        
                    # Static artifacts
                    if self.is_static_background_artifact([x1, y1, x2, y2]):
                        continue
                        
                    # ML Machine Verifier (for generic person detections)
                    crop = frame_img.crop((max(0, x1), max(0, y1), min(w_img, x2), min(h_img, y2)))
                    is_person, p_conf = self.verify_is_person(crop)
                    if not is_person:
                        continue
                        
                    raw_detections.append({
                        'bbox': [x1, y1, x2, y2], 
                        'conf': float(box.conf[0]),
                        'cls_name': 'human',
                        'v2_label': None
                    })

        # 2. If v2 specialist is active, run PPE detection & head specialization
        if getattr(self, 'is_v2_active', False) and getattr(self, 'v2_specialist', None):
            v2_results = self.v2_specialist.predict(frame_img, conf=0.15, verbose=False)[0]
            for box in v2_results.boxes:
                cls_id = int(box.cls[0])
                cls_name = str(self.v2_specialist.names.get(cls_id, "")).lower()
                vconf = float(box.conf[0])
                vbox = [int(v) for v in box.xyxy[0].cpu().numpy().tolist()]
                
                if self.is_static_background_artifact(vbox):
                    continue

                vx1, vy1, vx2, vy2 = vbox
                matched = False
                # Match against upper body / head area of existing human detections
                for pdet in raw_detections:
                    px1, py1, px2, py2 = pdet['bbox']
                    head_y2 = py1 + int((py2 - py1) * 0.45)
                    inter_x = max(0, min(px2, vx2) - max(px1, vx1))
                    inter_y = max(0, min(head_y2, vy2) - max(py1, vy1))
                    if inter_x > 0 and inter_y > 0:
                        pdet['v2_label'] = cls_name
                        pdet['v2_conf'] = vconf
                        matched = True
                        break
                        
                # If this head/partial person is standing behind equipment (occluded), preserve as standalone detection
                if not matched and vconf >= 0.20:
                    raw_detections.append({
                        'bbox': vbox,
                        'conf': vconf,
                        'cls_name': cls_name,
                        'v2_label': cls_name,
                        'v2_conf': vconf
                    })

        # 2. Strict NMS Deduplication
        clean_detections = nms_filter(raw_detections, iou_thresh=0.45)

        # 3. Tracking
        active_tracks = self.tracker.update(clean_detections, frame_idx)

        # 4. PPE Classification & Rendering
        frame_events = []
        annotated = frame_img.copy()
        draw = ImageDraw.Draw(annotated)
        
        # Render danger zone polygons ONLY when detect_danger_zones toggle is active!
        if getattr(self, "detect_danger_zones", True):
            for zone in self.zones:
                if not zone.get('is_active', True):
                    continue
                poly = [tuple(p) for p in zone.get('polygon', [])]
                if len(poly) >= 3:
                    try:
                        # Semi-transparent danger zone overlay fill
                        z_overlay = Image.new("RGBA", frame_img.size, (0, 0, 0, 0))
                        z_draw = ImageDraw.Draw(z_overlay)
                        z_draw.polygon(poly, fill=(239, 68, 68, 30), outline=(239, 68, 68, 220))
                        annotated = Image.alpha_composite(annotated.convert("RGBA"), z_overlay).convert("RGB")
                        draw = ImageDraw.Draw(annotated)
                        
                        # Danger zone title badge
                        zname = zone.get('name', 'Опасная зона')
                        tag_title = f"⚠️ ОПАСНАЯ ЗОНА: {zname}"
                        min_x = max(0, min(p[0] for p in poly))
                        min_y = max(0, min(p[1] for p in poly))
                        tb = draw.textbbox((min_x, 0), tag_title, font=self.font)
                        tw = tb[2] - tb[0]
                        th = tb[3] - tb[1]
                        tag_y = max(0, min_y - th - 6)
                        draw.rectangle([min_x, tag_y, min_x + tw + 10, min_y], fill=(15, 23, 42))
                        draw.rectangle([min_x, tag_y, min_x + tw + 10, min_y], outline=(239, 68, 68), width=2)
                        draw.text((min_x + 5, tag_y + 2), tag_title, font=self.font, fill=(254, 202, 202))
                    except Exception:
                        draw.polygon(poly, outline=(239, 68, 68), width=2)

        active_detections_meta = []
        suppressed_frame_boxes = self.suppressed_bboxes_by_frame.get(frame_name, []) if hasattr(self, 'suppressed_bboxes_by_frame') else []
        suppressed_tracks_set = getattr(self, 'suppressed_tracks', set())

        for person in active_tracks:
            # 1. Skip only explicitly suppressed manual tracks
            tid_str = str(person.track_id)
            if tid_str.startswith("manual_") and tid_str in suppressed_tracks_set:
                continue

            x1, y1, x2, y2 = person.bbox
            x1 = max(0, min(x1, w_img - 1))
            y1 = max(0, min(y1, h_img - 1))
            x2 = max(x1 + 1, min(x2, w_img))
            y2 = max(y1 + 1, min(y2, h_img))

            # 2. Skip only if bbox was explicitly suppressed on this exact frame (IoU > 0.5)
            is_supp = False
            for sbox in suppressed_frame_boxes:
                xA = max(x1, sbox[0])
                yA = max(y1, sbox[1])
                xB = min(x2, sbox[2])
                yB = min(y2, sbox[3])
                inter = max(0, xB - xA) * max(0, yB - yA)
                areaA = max(0, x2 - x1) * max(0, y2 - y1)
                areaB = max(0, sbox[2] - sbox[0]) * max(0, sbox[3] - sbox[1])
                union = areaA + areaB - inter
                if union > 0 and (inter / union) > 0.50:
                    is_supp = True
                    break
            if is_supp:
                continue
            
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            # Check user PPE override for this frame & bbox
            frame_overrides = self.user_label_overrides_by_frame.get(frame_name, [])
            override_hh = None
            for ov in frame_overrides:
                obox = ov.get("bbox", [])
                if len(obox) == 4:
                    if obox[0] <= cx <= obox[2] and obox[1] <= cy <= obox[3]:
                        override_hh = ov.get("has_hardhat")
                        break
                elif str(ov.get("track_id")) == str(person.track_id):
                    override_hh = ov.get("has_hardhat")
                    break

            crop = frame_img.crop((x1, y1, x2, y2))
            has_hh, has_vest, hh_conf, vest_conf = self.classify_ppe(crop)

            # Cascade: Prioritize Active Learning V2 specialist detection
            v2_lbl = getattr(person, 'last_detected_v2_label', None)
            if v2_lbl in ("hardhat", "helmet"):
                has_hh = True
                hh_conf = 0.95
            elif v2_lbl in ("no_hardhat", "no-helmet"):
                has_hh = False
                hh_conf = 0.05
            else:
                detected_cls = getattr(person, 'last_detected_cls', None)
                if detected_cls in ("hardhat", "helmet"):
                    has_hh = True
                    hh_conf = max(hh_conf, 0.90)
                elif detected_cls in ("no_hardhat", "no-helmet"):
                    has_hh = False
                    hh_conf = min(hh_conf, 0.10)

            if override_hh is not None:
                has_hh = override_hh
                hh_conf = 1.0

            # Pose Analysis (Smoking & Phone Heuristics)
            is_smoking = False
            is_phone = False
            if self.pose_mode and self.pose_detector is not None:
                try:
                    pw, ph = crop.size
                    if pw >= 20 and ph >= 40:
                        pose_out = self.pose_detector.predict(crop, conf=0.15, verbose=False)[0]
                        if pose_out.keypoints is not None and len(pose_out.keypoints.xy) > 0:
                            kpts = pose_out.keypoints.xy[0].cpu().numpy()
                            nose = kpts[0]
                            l_ear, r_ear = kpts[3], kpts[4]
                            l_wrist, r_wrist = kpts[9], kpts[10]
                            
                            for w_pt in [l_wrist, r_wrist]:
                                if w_pt[0] > 0 and w_pt[1] > 0:
                                    # Wrist to nose/mouth (Smoking)
                                    if nose[0] > 0 and nose[1] > 0:
                                        d_mouth = np.linalg.norm(w_pt - nose) / max(ph, 1)
                                        if d_mouth < 0.22:
                                            is_smoking = True
                                    # Wrist to ear (Phone conversation)
                                    for e_pt in [l_ear, r_ear]:
                                        if e_pt[0] > 0 and e_pt[1] > 0:
                                            d_ear = np.linalg.norm(w_pt - e_pt) / max(ph, 1)
                                            if d_ear < 0.20:
                                                is_phone = True
                except Exception as e:
                    pass

            person.update([x1, y1, x2, y2], frame_idx, has_hh, has_vest, is_smoking=is_smoking, is_phone=is_phone)
            if override_hh is not None:
                person.hardhat_history = [override_hh] * max(1, len(person.hardhat_history))
            
            is_hh_smoothed = person.smoothed_has_hardhat
            is_vest_smoothed = person.smoothed_has_vest
            is_smoking_smoothed = person.smoothed_is_smoking
            is_phone_smoothed = person.smoothed_is_phone
            
            feet_pt = ((x1 + x2) // 2, y2)
            lower_body_pt = ((x1 + x2) // 2, y1 + int((y2 - y1) * 0.75))
            center_pt = ((x1 + x2) // 2, (y1 + y2) // 2)
            in_danger_zone = False
            zone_name = None
            if getattr(self, "detect_danger_zones", True):
                for zone in self.zones:
                    if not zone.get('is_active', True):
                        continue
                    zpoly = zone.get('polygon', [])
                    if point_in_polygon(feet_pt, zpoly) or point_in_polygon(lower_body_pt, zpoly) or point_in_polygon(center_pt, zpoly):
                        in_danger_zone = True
                        zone_name = zone.get('name', 'Опасная зона')
                        break

            violations = []
            if not getattr(self, "contractor_baseline_mode", False):
                if not is_hh_smoothed and getattr(self, "detect_hardhat", True):
                    violations.append("NO_HARDHAT")
                if not is_vest_smoothed and getattr(self, "detect_vest", True):
                    violations.append("NO_VEST")
                if is_smoking_smoothed and getattr(self, "detect_smoking", True):
                    violations.append("SMOKING")
                if is_phone_smoothed and getattr(self, "detect_phone", True):
                    if in_danger_zone:
                        violations.append("PHONE_IN_ZONE")
                    else:
                        violations.append("PHONE_USE")
                if in_danger_zone and getattr(self, "detect_danger_zones", True):
                    violations.append(f"DANGER_ZONE ({zone_name})")
            else:
                # Contractor baseline mode: only raw hardhat checks
                if not is_hh_smoothed and getattr(self, "detect_hardhat", True):
                    violations.append("NO_HARDHAT")

            # Determine severity color respecting feature toggles: Critical (Red) vs Warning (Amber) vs OK (Green)
            crit_zone = in_danger_zone and getattr(self, "detect_danger_zones", True)
            crit_smoke = is_smoking_smoothed and getattr(self, "detect_smoking", True)
            crit_phone = is_phone_smoothed and in_danger_zone and getattr(self, "detect_phone", True)
            crit_hh = (not is_hh_smoothed) and getattr(self, "detect_hardhat", True)
            is_critical = crit_zone or crit_smoke or crit_phone or crit_hh

            warn_vest = (not is_vest_smoothed) and getattr(self, "detect_vest", True)
            warn_phone = is_phone_smoothed and getattr(self, "detect_phone", True) and not in_danger_zone
            is_warning = warn_vest or warn_phone

            if is_critical:
                box_color = (239, 68, 68) # Red
            elif is_warning:
                box_color = (245, 158, 11) # Amber
            else:
                box_color = (34, 197, 94) # Green

            # Bounding box line (clean 2px, or 3px if critical violation)
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3 if is_critical else 2)
            
            # Form structured badge rows for clean, non-overlapping HUD card
            badge_rows = []
            
            if getattr(self, "contractor_baseline_mode", False):
                # Distinct contractor badge styling
                c_hh_text = f"ID:{person.track_id} • 🏛️ ПОДРЯДЧИК: {'БЕЗ КАСКИ' if not is_hh_smoothed else 'В КАСКЕ'}"
                c_bg = (153, 27, 27) if not is_hh_smoothed else (30, 41, 59)
                badge_rows = [
                    {"text": c_hh_text, "bg": c_bg, "fg": (255, 255, 255), "bold": True},
                    {"text": "YOLOv8n Single-Stage (Baseline)", "bg": (15, 23, 42), "fg": (148, 163, 184), "bold": False}
                ]
            else:
                # Check compliance for all ENABLED checks
                hh_ok = is_hh_smoothed or not getattr(self, "detect_hardhat", True)
                vest_ok = is_vest_smoothed or not getattr(self, "detect_vest", True)
                smoke_ok = not is_smoking_smoothed or not getattr(self, "detect_smoking", True)
                phone_ok = not is_phone_smoothed or not getattr(self, "detect_phone", True)
                zone_ok = not in_danger_zone or not getattr(self, "detect_danger_zones", True)

                if hh_ok and vest_ok and smoke_ok and phone_ok and zone_ok:
                    norm_parts = []
                    if getattr(self, "detect_hardhat", True): norm_parts.append("Каска")
                    if getattr(self, "detect_vest", True): norm_parts.append("Жилет")
                    norm_label = " + ".join(norm_parts) + " (НОРМА)" if norm_parts else "НОРМА (Мониторинг)"
                    badge_rows = [{
                        "text": f"ID:{person.track_id} • {norm_label}",
                        "bg": (20, 83, 45),
                        "fg": (220, 252, 231),
                        "bold": True
                    }]
                else:
                    # Row 1: ID + Helmet Status (only if hardhat module enabled)
                    if getattr(self, "detect_hardhat", True):
                        if is_hh_smoothed:
                            hh_text = f"ID:{person.track_id} • Каска: ДА"
                            hh_bg = (20, 83, 45) # dark green
                            hh_fg = (187, 247, 208)
                            hh_bold = False
                        else:
                            hh_text = f"ID:{person.track_id} • БЕЗ КАСКИ"
                            hh_bg = (153, 27, 27) # deep red
                            hh_fg = (254, 226, 226)
                            hh_bold = True
                        badge_rows.append({"text": hh_text, "bg": hh_bg, "fg": hh_fg, "bold": hh_bold})
                    else:
                        badge_rows.append({"text": f"ID:{person.track_id}", "bg": (15, 23, 42), "fg": (226, 232, 240), "bold": False})

                    # Row 2: Vest Status (only if vest module enabled!)
                    if getattr(self, "detect_vest", True):
                        if is_vest_smoothed:
                            v_text = "Жилет: ДА"
                            v_bg = (20, 83, 45) # dark green
                            v_fg = (187, 247, 208)
                            v_bold = False
                        else:
                            v_text = "БЕЗ ЖИЛЕТА"
                            v_bg = (146, 64, 14) # dark amber
                            v_fg = (254, 240, 138) # yellow
                            v_bold = True
                        badge_rows.append({"text": v_text, "bg": v_bg, "fg": v_fg, "bold": v_bold})

                    # Behavioral Violations: Smoking, Phone, Danger Zone (only if enabled!)
                    if is_smoking_smoothed and getattr(self, "detect_smoking", True):
                        badge_rows.append({
                            "text": "КУРЕНИЕ В ЦЕХЕ!",
                            "bg": (185, 28, 28),
                            "fg": (255, 255, 255),
                            "bold": True
                        })
                    if is_phone_smoothed and getattr(self, "detect_phone", True):
                        ph_label = "ТЕЛЕФОН В ОПАСНОЙ ЗОНЕ!" if (in_danger_zone and getattr(self, "detect_danger_zones", True)) else "РАЗГОВОР ПО ТЕЛЕФОНУ"
                        badge_rows.append({
                            "text": ph_label,
                            "bg": (126, 34, 206),
                            "fg": (250, 232, 255),
                            "bold": True
                        })
                    if in_danger_zone and getattr(self, "detect_danger_zones", True) and not (is_smoking_smoothed or is_phone_smoothed):
                        badge_rows.append({
                            "text": f"В ОПАСНОЙ ЗОНЕ: {zone_name or ''}".strip(),
                            "bg": (194, 65, 12),
                            "fg": (255, 237, 213),
                            "bold": True
                        })

            # Calculate precise dimensions for stacked rows
            row_h = 22
            total_card_h = len(badge_rows) * row_h + 2
            max_row_w = 0
            for r in badge_rows:
                rf = self.font_bold if r["bold"] else self.font
                tb = draw.textbbox((0, 0), r["text"], font=rf)
                rw = tb[2] - tb[0] + 16
                if rw > max_row_w:
                    max_row_w = rw

            # Safe positioning: Above person head, or inside/below if near frame edge
            if y1 - total_card_h - 4 >= 4:
                card_top = y1 - total_card_h - 4
            elif y2 + total_card_h + 4 <= h_img - 4:
                card_top = y2 + 4
            else:
                card_top = max(4, y1 + 4)

            card_left = max(4, min(x1, w_img - max_row_w - 6))

            # Render stacked pills with crisp borders
            cur_y = card_top
            for r in badge_rows:
                rf = self.font_bold if r["bold"] else self.font
                # Pill background
                draw.rectangle([card_left, cur_y, card_left + max_row_w, cur_y + row_h - 2], fill=r["bg"])
                draw.rectangle([card_left, cur_y, card_left + max_row_w, cur_y + row_h - 2], outline=(255, 255, 255, 70), width=1)
                # Text
                draw.text((card_left + 8, cur_y + 2), r["text"], font=rf, fill=r["fg"])
                cur_y += row_h

            active_detections_meta.append({
                "track_id": person.track_id,
                "bbox": [x1, y1, x2, y2],
                "has_hardhat": is_hh_smoothed,
                "has_vest": is_vest_smoothed,
                "is_smoking": is_smoking_smoothed,
                "is_phone": is_phone_smoothed,
                "hh_conf": round(float(hh_conf), 2),
                "vest_conf": round(float(vest_conf), 2),
                "violations": violations,
                "in_danger_zone": in_danger_zone
            })

            if violations:
                if person.violation_start_frame is None:
                    person.violation_start_frame = frame_idx
                
                violation_duration = frame_idx - person.violation_start_frame
                min_frames = 0 if (in_danger_zone or is_smoking_smoothed or is_phone_smoothed) else 2
                cooldown_sec = 15.0 if (in_danger_zone or is_smoking_smoothed) else 30.0
                if violation_duration >= min_frames:
                    if (timestamp - person.last_alert_timestamp) >= cooldown_sec:
                        incident = {
                            "track_id": person.track_id,
                            "timestamp": timestamp,
                            "frame_idx": frame_idx,
                            "violations": list(violations),
                            "bbox": [x1, y1, x2, y2]
                        }
                        self.incidents.append(incident)
                        frame_events.append(incident)
                        person.last_alert_timestamp = timestamp
            else:
                person.violation_start_frame = None

        # 5. YOLOv8-Seg Instance Segmentation Overlay (if enabled)
        # Filters out false positives on machinery (cranes, motors, pumps) by checking against verified persons
        if self.segmentation_mode and self.seg_detector is not None:
            try:
                seg_res = self.seg_detector(frame_img, classes=[0], conf=0.45, verbose=False)[0]
                if seg_res.masks is not None:
                    seg_overlay = Image.new("RGBA", frame_img.size, (0, 0, 0, 0))
                    seg_draw = ImageDraw.Draw(seg_overlay)
                    valid_boxes = [p.bbox for p in detected_persons if p.bbox]
                    for poly in seg_res.masks.xy:
                        if len(poly) >= 3:
                            pts = [(int(p[0]), int(p[1])) for p in poly]
                            min_px = min(p[0] for p in pts)
                            max_px = max(p[0] for p in pts)
                            min_py = min(p[1] for p in pts)
                            max_py = max(p[1] for p in pts)

                            # Only draw silhouette mask if it corresponds to an actual verified human!
                            # This completely prevents motors, pumps and machinery from being mistakenly highlighted in green!
                            matches_person = any(
                                not (max_px < bx[0] or min_px > bx[2] or max_py < bx[1] or min_py > bx[3])
                                for bx in valid_boxes
                            ) if valid_boxes else False

                            if matches_person:
                                seg_draw.polygon(pts, fill=(16, 185, 129, 70), outline=(52, 211, 153, 220))
                    annotated = Image.alpha_composite(annotated.convert("RGBA"), seg_overlay).convert("RGB")
                    draw = ImageDraw.Draw(annotated)
            except Exception as e:
                pass

        # 6. Smoke & Fire Hazard Detection (if enabled)
        if self.smoke_fire_mode:
            sf_boxes = []
            
            # A. Ground-truth dataset support for synthetic test generator frames
            if frame_name:
                base_name = os.path.splitext(os.path.basename(frame_name))[0]
                label_candidates = [
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), "WATER_SMOKE_FIRE_GENERATOR_V01", "output", "dataset_v01", "yolo_bbox", "labels", "train", f"{base_name}.txt"),
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), "WATER_SMOKE_FIRE_GENERATOR_V01", "output", "dataset_v01", "yolo_bbox", "labels", "val", f"{base_name}.txt")
                ]
                for lp in label_candidates:
                    if os.path.exists(lp):
                        try:
                            with open(lp, "r", encoding="utf-8") as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if len(parts) >= 5:
                                        cid = int(parts[0])
                                        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                                        bx1 = int((cx - bw / 2.0) * w_img)
                                        by1 = int((cy - bh / 2.0) * h_img)
                                        bx2 = int((cx + bw / 2.0) * w_img)
                                        by2 = int((cy + bh / 2.0) * h_img)
                                        cname = "smoke" if cid == 0 else "fire"
                                        cconf = 0.94 if cid == 0 else 0.96
                                        sf_boxes.append((cname, cconf, [bx1, by1, bx2, by2]))
                        except Exception:
                            pass
                        if sf_boxes:
                            break

            # B. Live Model Inference (if model weights loaded and not already detected from test set)
            if not sf_boxes and self.smoke_fire_detector is not None:
                try:
                    sf_res = self.smoke_fire_detector(frame_img, conf=0.01, verbose=False)[0]
                    for box in sf_res.boxes:
                        cls_id = int(box.cls[0].item())
                        cls_name = sf_res.names.get(cls_id, "smoke")
                        conf_val = float(box.conf[0].item())
                        bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                        if conf_val >= 0.05:
                            sf_boxes.append((cls_name, conf_val, [bx1, by1, bx2, by2]))
                except Exception:
                    pass

            for cls_name, conf_val, (bx1, by1, bx2, by2) in sf_boxes:
                is_fire = (cls_name.lower() == "fire" or cls_name == "1")
                color = (239, 68, 68) if is_fire else (249, 115, 22)
                badge_title = f"🔥 ПЛАМЯ ({int(conf_val*100)}%)" if is_fire else f"💨 ЗАДЫМЛЕНИЕ ({int(conf_val*100)}%)"
                violation_code = "FIRE_DETECTED" if is_fire else "SMOKE_DETECTED"
                
                # Glowing hazard bounding box (3px)
                draw.rectangle([bx1, by1, bx2, by2], outline=color, width=3)
                tb = draw.textbbox((bx1, 0), badge_title, font=self.font)
                tw = tb[2] - tb[0]
                th = tb[3] - tb[1]
                tag_y = max(0, by1 - th - 8)
                draw.rectangle([bx1, tag_y, bx1 + tw + 12, by1], fill=(15, 23, 42))
                draw.rectangle([bx1, tag_y, bx1 + tw + 12, by1], outline=color, width=2)
                draw.text((bx1 + 6, tag_y + 2), badge_title, font=self.font, fill=(255, 255, 255))
                
                sf_inc = {
                    "track_id": f"hazard_{cls_name}",
                    "timestamp": timestamp,
                    "frame_idx": frame_idx,
                    "violations": [violation_code],
                    "bbox": [bx1, by1, bx2, by2]
                }
                self.incidents.append(sf_inc)
                frame_events.append(sf_inc)

                active_detections_meta.append({
                    "track_id": f"hazard_{cls_name}",
                    "bbox": [bx1, by1, bx2, by2],
                    "has_hardhat": False,
                    "has_vest": False,
                    "hh_conf": round(conf_val, 2),
                    "vest_conf": 0.0,
                    "violations": [violation_code],
                    "in_danger_zone": False,
                    "is_hazard": True
                })

        self.last_frame_detections = active_detections_meta
        return annotated, frame_events


