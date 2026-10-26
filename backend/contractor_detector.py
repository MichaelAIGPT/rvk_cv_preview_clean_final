import os
import cv2
import yaml
import numpy as np
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "RVK-AI-Vision-current-runtime-contractor-20260928", "model", "construction-ppe-960", "model.onnx")
META_PATH = os.path.join(BASE_DIR, "RVK-AI-Vision-current-runtime-contractor-20260928", "model", "construction-ppe-960", "metadata.yaml")

class ContractorONNXDetector:
    def __init__(self):
        self.session = None
        self.names = {}
        self.target_size = 960
        self._load_model()

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            print(f"[ContractorDetector] Warning: Model not found at {MODEL_PATH}")
            return

        try:
            # Load metadata
            if os.path.exists(META_PATH):
                with open(META_PATH, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    self.names = data.get("names", {})
            if not self.names:
                self.names = {
                    0: "Hardhat", 1: "Mask", 2: "NO-Hardhat", 3: "NO-Mask",
                    4: "NO-Safety Vest", 5: "Person", 6: "Safety Cone",
                    7: "Safety Vest", 8: "machinery", 9: "vehicle"
                }

            # Optimized CPU execution provider
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"])
            print(f"[ContractorDetector] Loaded real contractor ONNX model: {MODEL_PATH} (960x960, 10 classes)")
        except Exception as e:
            print(f"[ContractorDetector] Error loading ONNX model: {e}")

    def detect(self, img_bgr: np.ndarray, conf_thresh: float = 0.22, iou_thresh: float = 0.45) -> list:
        if self.session is None or img_bgr is None:
            return []

        h, w = img_bgr.shape[:2]
        scale = min(self.target_size / w, self.target_size / h)
        rw, rh = round(w * scale), round(h * scale)
        resized = cv2.resize(img_bgr, (rw, rh))

        canvas = np.full((self.target_size, self.target_size, 3), 114, dtype=np.uint8)
        dx = (self.target_size - rw) // 2
        dy = (self.target_size - rh) // 2
        canvas[dy:dy+rh, dx:dx+rw] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.transpose(rgb, (2, 0, 1))[None]

        try:
            outputs = self.session.run(["output0"], {"images": tensor})[0]
            pred = outputs[0]
            if pred.shape[0] < pred.shape[1]:
                pred = pred.T  # (18900, 14)

            boxes = pred[:, :4]
            scores = pred[:, 4:]
            class_ids = np.argmax(scores, axis=1)
            confidences = np.max(scores, axis=1)

            mask = confidences >= conf_thresh
            filtered_boxes = boxes[mask]
            filtered_confs = confidences[mask]
            filtered_classes = class_ids[mask]

            if len(filtered_confs) == 0:
                return []

            nms_boxes = []
            for (cx, cy, bw, bh) in filtered_boxes:
                x1 = int((cx - bw / 2 - dx) / scale)
                y1 = int((cy - bh / 2 - dy) / scale)
                x2 = int((cx + bw / 2 - dx) / scale)
                y2 = int((cy + bh / 2 - dy) / scale)
                # clamp to image boundaries
                x1 = max(0, min(w, x1))
                y1 = max(0, min(h, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))
                nms_boxes.append([x1, y1, max(1, x2 - x1), max(1, y2 - y1)])

            indices = cv2.dnn.NMSBoxes(nms_boxes, filtered_confs.tolist(), conf_thresh, iou_thresh)
            results = []
            if len(indices) > 0:
                for idx in indices:
                    i = idx if isinstance(idx, (int, np.integer)) else idx[0]
                    cid = int(filtered_classes[i])
                    cname = self.names.get(cid, f"class_{cid}")
                    conf = float(filtered_confs[i])
                    b = nms_boxes[i]
                    x1, y1, bw, bh = b
                    results.append({
                        "bbox": [x1, y1, x1 + bw, y1 + bh],
                        "class_id": cid,
                        "class_name": cname,
                        "conf": round(conf, 3),
                        "model": "contractor_baseline"
                    })
            return results
        except Exception as e:
            print(f"[ContractorDetector] Inference error: {e}")
            return []

contractor_detector = ContractorONNXDetector()
