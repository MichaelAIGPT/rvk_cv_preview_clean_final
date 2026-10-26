import numpy as np

def calculate_iou(boxA, boxB):
    """
    Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    union = float(boxAArea + boxBArea - interArea)
    return interArea / union if union > 0 else 0.0

class TrackedPerson:
    def __init__(self, track_id, bbox, frame_idx):
        self.track_id = track_id
        self.bbox = bbox # [x1, y1, x2, y2]
        self.last_frame = frame_idx
        self.missed_frames = 0
        
        # Temporal smoothing history
        self.hardhat_history = []
        self.vest_history = []
        self.smoking_history = []
        self.phone_history = []
        
        # Violation tracking
        self.violation_start_frame = None
        self.last_alert_timestamp = 0.0

    def update(self, bbox, frame_idx, has_hardhat, has_vest, is_smoking=False, is_phone=False):
        self.bbox = bbox
        self.last_frame = frame_idx
        self.missed_frames = 0
        
        # Keep sliding window of last 10 frames
        self.hardhat_history.append(has_hardhat)
        if len(self.hardhat_history) > 10:
            self.hardhat_history.pop(0)
            
        self.vest_history.append(has_vest)
        if len(self.vest_history) > 10:
            self.vest_history.pop(0)

        self.smoking_history.append(bool(is_smoking))
        if len(self.smoking_history) > 10:
            self.smoking_history.pop(0)

        self.phone_history.append(bool(is_phone))
        if len(self.phone_history) > 10:
            self.phone_history.pop(0)

    @property
    def smoothed_has_hardhat(self):
        if not self.hardhat_history:
            return True
        return (sum(self.hardhat_history) / len(self.hardhat_history)) >= 0.5

    @property
    def smoothed_has_vest(self):
        if not self.vest_history:
            return False
        return (sum(self.vest_history) / len(self.vest_history)) >= 0.5

    @property
    def smoothed_is_smoking(self):
        if not self.smoking_history:
            return False
        return bool(self.smoking_history[-1]) or ((sum(self.smoking_history) / len(self.smoking_history)) >= 0.3)

    @property
    def smoothed_is_phone(self):
        if not self.phone_history:
            return False
        return bool(self.phone_history[-1]) or ((sum(self.phone_history) / len(self.phone_history)) >= 0.3)

class BipartiteTracker:
    """
    Robust Bipartite IoU Tracker resolving the critical bugs in Pavel's script:
    1. Supports adding new tracks across all frames (not just frame 1).
    2. Implements track memory (max_missed_frames=15) to survive temporary occlusions.
    3. Uses strict IoU threshold (min_iou=0.25) to prevent identity switches and jumps.
    4. Avoids naive 50/50 bounding box slicing.
    """
    def __init__(self, min_iou=0.25, max_missed_frames=15):
        self.min_iou = min_iou
        self.max_missed_frames = max_missed_frames
        self.next_id = 1
        self.tracks = {} # track_id -> TrackedPerson

    def update(self, detections, frame_idx):
        """
        detections: list of dicts: [{'bbox': [x1, y1, x2, y2], 'confidence': 0.85, ...}]
        returns: list of updated TrackedPerson objects active in this frame
        """
        matched_tracks = set()
        matched_dets = set()
        
        # Compute IoU cost matrix between existing active tracks and new detections
        track_ids = list(self.tracks.keys())
        if track_ids and detections:
            iou_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
            for i, tid in enumerate(track_ids):
                for j, det in enumerate(detections):
                    iou_matrix[i, j] = calculate_iou(self.tracks[tid].bbox, det['bbox'])
                    
            # Greedy bipartite matching by highest IoU
            while True:
                max_val = np.max(iou_matrix)
                if max_val < self.min_iou:
                    break
                i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                tid = track_ids[i]
                
                # Update matched track (classification will be applied by caller)
                self.tracks[tid].bbox = detections[j]['bbox']
                self.tracks[tid].last_frame = frame_idx
                self.tracks[tid].missed_frames = 0
                self.tracks[tid].last_detected_cls = detections[j].get('cls_name', None)
                self.tracks[tid].last_detected_v2_label = detections[j].get('v2_label', None)
                
                matched_tracks.add(tid)
                matched_dets.add(j)
                
                # Invalidate matched row and column
                iou_matrix[i, :] = -1.0
                iou_matrix[:, j] = -1.0

        # Handle unmatched detections -> create new tracks
        for j, det in enumerate(detections):
            if j not in matched_dets:
                new_person = TrackedPerson(self.next_id, det['bbox'], frame_idx)
                new_person.last_detected_cls = det.get('cls_name', None)
                new_person.last_detected_v2_label = det.get('v2_label', None)
                self.tracks[self.next_id] = new_person
                matched_tracks.add(self.next_id)
                self.next_id += 1

        # Handle unmatched existing tracks -> increment missed_frames or delete expired
        dead_tracks = []
        for tid, track in self.tracks.items():
            if tid not in matched_tracks:
                track.missed_frames += 1
                if track.missed_frames > self.max_missed_frames:
                    dead_tracks.append(tid)

        for tid in dead_tracks:
            del self.tracks[tid]

        # Return tracks that were observed in this exact frame
        return [self.tracks[tid] for tid in matched_tracks]
