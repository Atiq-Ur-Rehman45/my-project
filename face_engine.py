
import cv2
import numpy as np
import pickle
import os
import time
import logging
from collections import defaultdict, deque

from config import *

logger = logging.getLogger(__name__)

class FaceEngine:
    """
    Production-grade single engine:
    - SFace (YuNet + SFace DNN)
    """

    def __init__(self):
        self.engine_type = "SFACE"
        self.label_map = {}
        self.model_loaded = False
        self._sface_input_size = None
        self._sface_labels = np.empty((0,), dtype=np.int32)
        self._sface_embedding_matrix = np.empty((0, 0), dtype=np.float32)
        self._sface_frame_index = 0
        self._sface_track_cache = []
        self._sface_track_seq = 0
        
        print(f"\n[ENGINE] =======================================")
        print(f"[ENGINE] Booting: SFACE Recognition Engine")
        print(f"[ENGINE] =======================================")
        
        self._init_sface()
        
        print(f"[ENGINE] OK: Engine ready\n")

    # ══════════════════════════════════════════════════════════════════════════
    # INITIALIZATION
    # ══════════════════════════════════════════════════════════════════════════

    def _init_sface(self):
        """Initialize Deep Learning YuNet + SFace."""
        try:
            self.detector = cv2.FaceDetectorYN.create(
                YUNET_MODEL_PATH, "", YUNET_INPUT_SIZE,
                score_threshold=YUNET_SCORE_THRESHOLD,
                nms_threshold=YUNET_NMS_THRESHOLD, 
                top_k=YUNET_TOP_K
            )
            self.recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL_PATH, "")
            self.embeddings_db = {}
            self._set_sface_input_size((FRAME_WIDTH, FRAME_HEIGHT))
            
            if os.path.exists(SFACE_DB_PATH):
                with open(SFACE_DB_PATH, 'rb') as f:
                    data = pickle.load(f)
                    self.embeddings_db = data.get('embeddings', {})
                    self.model_loaded = len(self.embeddings_db) > 0
                    self._refresh_sface_index()
                    print(f"[ENGINE] OK: Loaded {len(self.embeddings_db)} SFace profiles")
            else:
                self._refresh_sface_index()
                print("[ENGINE] No SFace database found (will create on first enrollment)")
                
        except Exception as e:
            print(f"[ENGINE] ERROR: SFace initialization failed: {e}")

    def _set_sface_input_size(self, size):
        """Update the detector input size only when dimensions change."""
        if self._sface_input_size != size:
            self.detector.setInputSize(size)
            self._sface_input_size = size

    def _normalize_sface_embedding(self, feature):
        """Normalize embeddings so vectorized dot products match cosine similarity."""
        vector = np.asarray(feature, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    def _apply_usm_sharpening(self, img, amount=0.8, sigma=1.0):
        """Tier 2 Fix 3: Apply Unsharp Mask (USM) filter to small/distant faces."""
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
        return sharpened

    def _refresh_sface_index(self):
        """Build a contiguous embedding matrix — one row per stored embedding.
        Fix 1: Supports multiple embeddings per label for angle-diversity matching."""
        labels = []
        vectors = []

        for label, stored in self.embeddings_db.items():
            # Backward-compatible: old db has single ndarray, new db has list of ndarrays
            embeddings = stored if isinstance(stored, list) else [stored]
            for emb in embeddings:
                labels.append(int(label))
                vectors.append(self._normalize_sface_embedding(emb))

        if vectors:
            self._sface_labels = np.asarray(labels, dtype=np.int32)
            self._sface_embedding_matrix = np.vstack(vectors).astype(np.float32)
        else:
            self._sface_labels = np.empty((0,), dtype=np.int32)
            self._sface_embedding_matrix = np.empty((0, 0), dtype=np.float32)


    def _bbox_iou(self, box_a, box_b):
        """Compute IoU between two (x, y, w, h) boxes."""
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b

        a_x2 = ax + aw
        a_y2 = ay + ah
        b_x2 = bx + bw
        b_y2 = by + bh

        ix1 = max(ax, bx)
        iy1 = max(ay, by)
        ix2 = min(a_x2, b_x2)
        iy2 = min(a_y2, b_y2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih

        area_a = max(0, aw) * max(0, ah)
        area_b = max(0, bw) * max(0, bh)
        union = area_a + area_b - inter

        if union <= 0:
            return 0.0
        return inter / union

    def _prune_sface_cache(self, now_ts):
        """Keep only fresh tracks and cap cache size."""
        self._sface_track_cache = [
            track for track in self._sface_track_cache
            if now_ts - track["last_seen_ts"] <= SFACE_CACHE_TTL_SECONDS
        ]

        if len(self._sface_track_cache) > SFACE_CACHE_MAX_TRACKS:
            self._sface_track_cache.sort(key=lambda t: t["last_seen_ts"], reverse=True)
            self._sface_track_cache = self._sface_track_cache[:SFACE_CACHE_MAX_TRACKS]

    def _find_cache_track(self, box):
        """Find the best cached track for a detected box."""
        best_idx = -1
        best_iou = 0.0

        for idx, track in enumerate(self._sface_track_cache):
            iou = self._bbox_iou(box, track["box"])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_iou >= SFACE_CACHE_IOU_THRESHOLD:
            return best_idx
        return -1

    def _sface_quality_gate(self, frame, box, blur_threshold=None):
        """Reject low-quality detections before identity matching."""
        if blur_threshold is None:
            blur_threshold = SFACE_RECOG_BLUR_THRESHOLD
        x, y, w, h = box
        frame_area = float(max(1, frame.shape[0] * frame.shape[1]))
        face_area_ratio = (w * h) / frame_area
        if face_area_ratio < SFACE_RECOG_MIN_FACE_AREA_RATIO:
            return False, "face_too_small"

        y1 = max(0, y)
        y2 = min(frame.shape[0], y + h)
        x1 = max(0, x)
        x2 = min(frame.shape[1], x + w)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False, "face_out_of_frame"

        if ENABLE_BLUR_DETECTION:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur_score = self.estimate_blur(roi_gray)
            if blur_score < blur_threshold:
                return False, "blurry"

        return True, "ok"

    def _run_sface_match(self, frame, face):
        """Compute recognition scores using multi-embedding per-label matching (Fix 1)."""
        best_match_label = -1
        best_score = 0.0
        second_score = 0.0
        score_margin = 0.0
        name = UNKNOWN_LABEL

        if self.model_loaded and self._sface_embedding_matrix.size > 0:
            try:
                # Elite Fix 3: Sub-pixel Landmark Precision
                face_f32 = np.asarray(face, dtype=np.float32)
                aligned_face = self.recognizer.alignCrop(frame, face_f32)
                feature = self._normalize_sface_embedding(
                    self.recognizer.feature(aligned_face)
                )
                scores = self._sface_embedding_matrix @ feature

                if scores.size > 0:
                    # Elite Fix 1: Robust Matching via Top-K Mean Similarity
                    # Instead of max(), average the best matches per person to reduce false outliers.
                    label_scores = defaultdict(list)
                    for i, lbl in enumerate(self._sface_labels):
                        label_scores[int(lbl)].append(float(scores[i]))

                    label_results = {}
                    for lbl, score_list in label_scores.items():
                        # Sort descending and take top N for robust mean
                        top_scores = sorted(score_list, reverse=True)[:SFACE_TOP_K_MATCH]
                        label_results[lbl] = sum(top_scores) / len(top_scores)

                    sorted_labels = sorted(label_results.items(), key=lambda x: x[1], reverse=True)
                    best_match_label = sorted_labels[0][0]
                    best_score = sorted_labels[0][1]
                    if len(sorted_labels) > 1:
                        second_score = sorted_labels[1][1]
                    score_margin = best_score - second_score

                if best_match_label != -1:
                    criminal = self.label_map.get(best_match_label)
                    name = criminal["name"] if criminal else f"ID_{best_match_label}"

            except Exception as e:
                logger.warning(f"[ENGINE] SFace recognition error: {e}")

        return best_match_label, best_score, second_score, score_margin, name



    # ══════════════════════════════════════════════════════════════════════════
    # DETECTION & RECOGNITION (SFACE)
    # ══════════════════════════════════════════════════════════════════════════

    def _recognize_sface(self, frame):
        """SFace detection and recognition with quality, margin, and temporal gating."""
        self._sface_frame_index += 1
        height, width = frame.shape[:2]
        self._set_sface_input_size((width, height))

        # Fix 6: Normalize lighting via CLAHE — use normalized copy for detection/recognition,
        # keep original frame for drawing overlays
        frame_norm = self._apply_clahe_normalization(frame)

        _, faces = self.detector.detect(frame_norm)

        results = []
        if faces is None:
            return results, frame

        now_ts = time.time()
        self._prune_sface_cache(now_ts)
        
        for face in faces:
            x, y, w, h = face[0:4].astype(int)
            box = (int(x), int(y), int(w), int(h))

            cache_idx = self._find_cache_track(box)

            if cache_idx != -1:
                track = self._sface_track_cache[cache_idx]
                track["box"] = box
                track["last_seen_ts"] = now_ts
                # ── Frame-skip: reuse cached result if recognized recently ──
                frames_since = self._sface_frame_index - track.get("last_recognition_frame", 0)
                if frames_since < SFACE_SKIP_RECOGNITION_FRAMES and "last_result" in track:
                    cached_r = track["last_result"]
                    results.append({
                        "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                        "label": cached_r["label"],
                        "confidence": cached_r["confidence"],
                        "name": cached_r["name"],
                        "criminal": cached_r["criminal"],
                        "is_known": cached_r["is_known"],
                        "score_margin": cached_r.get("score_margin", 0.0),
                        "second_score": cached_r.get("second_score", 0.0),
                        "quality_ok": cached_r.get("quality_ok", True),
                        "quality_reason": "cached",
                        "consensus_count": cached_r.get("consensus_count", 0),
                        "was_sharpened": cached_r.get("was_sharpened", False),
                    })
                    continue
            else:
                self._sface_track_seq += 1
                track = {
                    "track_id": self._sface_track_seq,
                    "box": box,
                    "last_seen_ts": now_ts,
                    "last_recognition_frame": self._sface_frame_index,
                    "candidate_history": deque(maxlen=SFACE_CONSENSUS_WINDOW), # Stores (label, confidence)
                    "known_label": -1,
                    "known_name": UNKNOWN_LABEL,
                    "hold_until_frame": 0,
                }
                self._sface_track_cache.append(track)

            # Tier 2 Fix 3: Adaptive Small-Face Sharpening (with Hysteresis)
            # Apply USM if face is distant/small to recover edge contrast for alignment.
            # Hysteresis prevents the indicator from blinking at the threshold distance.
            prev_sharpened = track.get("was_sharpened", False)
            threshold = SFACE_SHARPEN_MIN_SIZE
            if prev_sharpened:
                threshold += 8  # Hysteresis buffer (10%)
            
            was_sharpened = (w < threshold or h < threshold)
            track["was_sharpened"] = was_sharpened # Persist state in track
            
            if was_sharpened:
                # Re-normalize base from original frame then sharpen to avoid double-CLAHE artifacts
                frame_norm = self._apply_clahe_normalization(frame) 
                frame_norm = self._apply_usm_sharpening(frame_norm)

            best_match_label, best_score, second_score, score_margin, matched_name = self._run_sface_match(frame_norm, face)

            # Strict quality for initial identification, lenient for keeping it during motion
            quality_acquire, q_reason_acq = self._sface_quality_gate(frame, box)
            quality_maintain, q_reason_mnt = self._sface_quality_gate(frame, box, blur_threshold=SFACE_RECOG_BLUR_THRESHOLD_MAINTAIN)
            
            consensus_count = 0

            # Hysteresis: stricter thresholds to acquire identity, softer thresholds to maintain it.
            acquiring_score_ok = best_score >= SFACE_MATCH_THRESHOLD_ACQUIRE
            acquiring_margin_ok = score_margin >= SFACE_MARGIN_THRESHOLD_ACQUIRE
            maintaining_same_label = (
                track["known_label"] != -1 and best_match_label == track["known_label"]
            )
            maintaining_score_ok = best_score >= SFACE_MATCH_THRESHOLD_MAINTAIN
            maintaining_margin_ok = score_margin >= SFACE_MARGIN_THRESHOLD_MAINTAIN

            candidate_acquire = (
                quality_acquire
                and best_match_label != -1
                and best_score >= SFACE_MATCH_FLOOR
                and acquiring_score_ok
                and acquiring_margin_ok
            )
            candidate_maintain = (
                quality_maintain
                and maintaining_same_label
                and maintaining_score_ok
                and maintaining_margin_ok
            )

            # Tier 2 Fix 2: Weighted Temporal Consensus
            # Store (label, score) in history for exponential decay weighting
            track["candidate_history"].append((int(best_match_label) if candidate_acquire else -1, best_score))
            
            # Calculate weighted consensus and weighted average score
            weighted_score = best_score
            if len(track["candidate_history"]) > 0:
                total_weight = 0.0
                sum_weighted_score = 0.0
                match_weight = 0.0
                
                # Exponential decay: weight = 0.9^n (n=0 is newest)
                # Iterate history from newest to oldest
                for i, (hist_label, hist_score) in enumerate(reversed(track["candidate_history"])):
                    weight = 0.9 ** i
                    total_weight += weight
                    sum_weighted_score += hist_score * weight
                    if hist_label != -1 and hist_label == best_match_label:
                        match_weight += weight
                
                weighted_score = sum_weighted_score / total_weight
                consensus_count = match_weight # Use sum of weights as 'count' for gating

            just_acquired = candidate_acquire and consensus_count >= SFACE_CONSENSUS_FRAMES
            if just_acquired:
                track["known_label"] = int(best_match_label)
                track["known_name"] = matched_name
                track["hold_until_frame"] = self._sface_frame_index + SFACE_KNOWN_HOLD_FRAMES

            hold_active = (
                track["known_label"] != -1
                and self._sface_frame_index <= track["hold_until_frame"]
                and best_match_label in (-1, track["known_label"])
            )

            is_known = just_acquired or candidate_maintain or hold_active

            if is_known and track["known_label"] != -1:
                final_label = int(track["known_label"])
                final_name = track["known_name"]
                criminal_data = self.label_map.get(final_label)
                if candidate_maintain:
                    track["hold_until_frame"] = self._sface_frame_index + SFACE_KNOWN_HOLD_FRAMES
            else:
                final_label = -1
                final_name = UNKNOWN_LABEL
                criminal_data = None
                if self._sface_frame_index > track["hold_until_frame"]:
                    track["known_label"] = -1
                    track["known_name"] = UNKNOWN_LABEL

            track["last_recognition_frame"] = self._sface_frame_index
            
            result_entry = {
                "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                "label": final_label,
                "confidence": float(weighted_score),
                "name": final_name,
                "criminal": criminal_data,
                "is_known": is_known,
                "score_margin": float(score_margin),
                "second_score": float(second_score),
                "quality_ok": quality_acquire if not is_known else quality_maintain,
                "quality_reason": q_reason_acq if not is_known else q_reason_mnt,
                "consensus_count": float(consensus_count),
                "was_sharpened": was_sharpened,
            }
            track["last_result"] = result_entry
            results.append(result_entry)
        
        return results, frame


    # ══════════════════════════════════════════════════════════════════════════
    # MAIN RECOGNITION ROUTER
    # ══════════════════════════════════════════════════════════════════════════

    def recognize_all_faces(self, frame):
        """Recognize faces using SFace."""
        return self._recognize_sface(frame)

    # ══════════════════════════════════════════════════════════════════════════
    # TRAINING (SFACE)
    # ══════════════════════════════════════════════════════════════════════════

    def _train_sface(self, training_data):
        """
        Train SFace: stores all individual embeddings per person (not averaged).
        Applies CLAHE lighting normalization before embedding extraction.
        Fix 1 (multi-embedding) + Fix 2 (CLAHE on training images).
        """
        print(f"[ENGINE] Training SFace with {len(training_data)} images...")

        embeddings_by_label = defaultdict(list)

        for img, label in training_data:
            # Fix 2: Apply CLAHE normalization before embedding extraction
            img_norm = self._apply_clahe_normalization(img)
            height, width = img_norm.shape[:2]
            self._set_sface_input_size((width, height))

            _, faces = self.detector.detect(img_norm)

            if faces is None or len(faces) == 0:
                logger.warning(f"[ENGINE] No face detected in training image for label {label}")
                continue

            # Use largest face
            faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            face = faces_sorted[0]

            try:
                # Elite Fix 3: Sub-pixel Landmark Precision during training
                face_f32 = np.asarray(face, dtype=np.float32)
                aligned_face = self.recognizer.alignCrop(img_norm, face_f32)
                # Fix 1: Store each embedding individually (pre-normalized)
                feature = self._normalize_sface_embedding(self.recognizer.feature(aligned_face))
                embeddings_by_label[label].append(feature)
            except Exception as e:
                logger.warning(f"[ENGINE] Failed to extract feature for label {label}: {e}")

        if not embeddings_by_label:
            print("[ENGINE] ✗ No valid faces found in training data")
            return False

        # Tier 2 Fix 1: Automatic Outlier Filtering
        self._filter_enrollment_outliers(embeddings_by_label)

        # Fix 1: Store list of all embeddings per label — never average them
        rebuilt_db = {}
        for label, features in embeddings_by_label.items():
            rebuilt_db[label] = features  # list of normalized numpy vectors
            print(f"  [+] Label {label}: {len(features)} embeddings stored (angle-diverse)")

        self.embeddings_db = rebuilt_db
        self._refresh_sface_index()

        with open(SFACE_DB_PATH, 'wb') as f:
            pickle.dump({'embeddings': self.embeddings_db}, f)

        self.model_loaded = True
        total_embs = sum(len(v) for v in rebuilt_db.values())
        print(f"[ENGINE] OK: SFace training complete: {len(self.embeddings_db)} persons, {total_embs} total embeddings")
        print(f"[ENGINE] OK: Database saved to {SFACE_DB_PATH}")
        return True

    def _filter_enrollment_outliers(self, embs_dict):
        """Tier 2 Fix 1: Discard embeddings that deviate too far from the person's average."""
        for label, embs in embs_dict.items():
            if len(embs) < SFACE_OUTLIER_MIN_SAMPLES:
                continue

            # Compute pairwise similarity matrix (normalized vectors means dot product = cosine similarity)
            matrix = np.vstack(embs)
            sim_matrix = matrix @ matrix.T
            
            # Distance = 1 - similarity
            dist_matrix = 1.0 - sim_matrix
            
            # Average distance for each sample (excluding self-distance which is 0)
            n = len(embs)
            avg_distances = (np.sum(dist_matrix, axis=1)) / (n - 1)
            
            # Identify outliers
            cleaned = []
            discarded_count = 0
            for i, dist in enumerate(avg_distances):
                # Discard if distance is high, but safeguard the minimum sample count
                if dist > SFACE_OUTLIER_THRESHOLD and (len(embs) - discarded_count) > SFACE_OUTLIER_MIN_SAMPLES:
                    discarded_count += 1
                    continue
                cleaned.append(embs[i])
            
            if discarded_count > 0:
                print(f"  [!] Label {label}: Discarded {discarded_count} outlier embeddings (Threshold: {SFACE_OUTLIER_THRESHOLD})")
                embs_dict[label] = cleaned


    # ══════════════════════════════════════════════════════════════════════════
    # UNIFIED TRAINING INTERFACE
    # ══════════════════════════════════════════════════════════════════════════

    def train(self, training_data):
        """Trains the SFace engine on the provided data."""
        return self._train_sface(training_data)

    # ══════════════════════════════════════════════════════════════════════════
    # ENROLLMENT (Face Sample Collection)
    # ══════════════════════════════════════════════════════════════════════════

    def _sface_pose_signature(self, face):
        """Estimate coarse yaw/pitch from YuNet 5-point landmarks."""
        if len(face) < 14:
            return None, None

        left_eye = np.array([face[4], face[5]], dtype=np.float32)
        right_eye = np.array([face[6], face[7]], dtype=np.float32)
        nose = np.array([face[8], face[9]], dtype=np.float32)
        mouth_left = np.array([face[10], face[11]], dtype=np.float32)
        mouth_right = np.array([face[12], face[13]], dtype=np.float32)

        eye_dx = right_eye[0] - left_eye[0]
        if abs(float(eye_dx)) < 1e-6:
            return None, None

        # Yaw proxy: nose position between eyes. Front is near 0.5.
        yaw = float((nose[0] - left_eye[0]) / eye_dx)

        eye_mid_y = float((left_eye[1] + right_eye[1]) * 0.5)
        mouth_mid_y = float((mouth_left[1] + mouth_right[1]) * 0.5)
        vertical_span = max(mouth_mid_y - eye_mid_y, 1e-6)

        # Pitch proxy: nose vertical offset between eye-line and mouth-line.
        pitch = float((nose[1] - eye_mid_y) / vertical_span)
        return yaw, pitch

    def _pose_matches_stage(self, angle, yaw, pitch, relaxed=False):
        """Validate coarse face orientation for the requested enrollment stage."""
        if yaw is None or pitch is None:
            return False

        angle = (angle or "").upper()
        if relaxed:
            if angle == "FRONT":
                return 0.38 <= yaw <= 0.62 and 0.34 <= pitch <= 0.76
            if angle == "LEFT":
                return yaw <= 0.49
            if angle == "RIGHT":
                return yaw >= 0.51
            if angle == "UP":
                return pitch <= 0.49
            if angle == "DOWN":
                return pitch >= 0.59

        if angle == "FRONT":
            return 0.42 <= yaw <= 0.58 and 0.40 <= pitch <= 0.70
        if angle == "LEFT":
            return yaw <= 0.46
        if angle == "RIGHT":
            return yaw >= 0.54
        if angle == "UP":
            return pitch <= 0.44
        if angle == "DOWN":
            return pitch >= 0.64
        return True

    def _liveness_pose_delta_ok(self, angle, yaw, pitch, baseline):
        """Check that current stage shows real head movement from a front baseline."""
        if baseline is None or yaw is None or pitch is None:
            return False

        base_yaw = baseline.get("yaw")
        base_pitch = baseline.get("pitch")
        if base_yaw is None or base_pitch is None:
            return False

        angle = (angle or "").upper()
        delta = ENROLL_REQUIRED_POSE_DELTA

        if angle == "LEFT":
            return (base_yaw - yaw) >= delta
        if angle == "RIGHT":
            return (yaw - base_yaw) >= delta
        if angle == "UP":
            return (base_pitch - pitch) >= delta
        if angle == "DOWN":
            return (pitch - base_pitch) >= delta
        return True

    def collect_face_samples(self, camera_index=0, target_count=None, label=0, save_dir=None):
        """
        Collect face samples from webcam with guided UI prompts and non-blocking pauses.
        """
        if target_count is None:
            target_count = ENROLL_FRAME_COUNT
        
        strategy = ENROLLMENT_STRATEGY
        total_target = sum(s["count"] for s in strategy)
        
        # ── 1. FORCE DIRECTSHOW (Bypass MSMF Timeout) ──
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)      
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     
    
        # ── 2. TIME-BASED WARMUP (Fixes the read-loop freeze) ──
        print("[ENGINE] Warming up camera...")
        warmup_start = time.time()
        while time.time() - warmup_start < 2.0:
            cap.read()
        
        if not cap.isOpened():
            raise RuntimeError(f"[ENGINE] Cannot open camera {camera_index}")
        
        collected = []
        current_angle_idx = 0
        angle_count = 0
        last_capture = 0
        capture_delay = ENROLL_CAPTURE_DELAY
        
        print(f"\n[ENGINE] ===========================================")
        print(f"[ENGINE]   ENROLLMENT MODE: {self.engine_type}")
        print(f"[ENGINE] ===========================================")
        print(f"[ENGINE] Target: {total_target} images across {len(strategy)} angles")
        print(f"[ENGINE] Press SPACE to capture, Q to cancel\n")
        
        input("[ENGINE] Press ENTER to begin...")
        
        # Initial countdown
        countdown_start = time.time()
        while time.time() - countdown_start < ENROLL_COUNTDOWN:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            display = frame.copy()
            
            remaining = ENROLL_COUNTDOWN - int(time.time() - countdown_start)
            cv2.putText(display, str(remaining), 
                       (FRAME_WIDTH//2 - 60, FRAME_HEIGHT//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 6.0, COLOR_ORANGE, 12)
            
            cv2.imshow("Enrollment", display)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                cap.release()
                cv2.destroyAllWindows()
                return []
        
        # Main collection loop variables
        total_collected = 0
        pause_until = 0  # Controls the non-blocking UI pause
        stage_start_time = time.time()
        stage_stable_frames = 0
        front_pose_baseline = None
        stage_metrics = [
            {
                "angle": stage.get("angle", f"STAGE_{idx + 1}"),
                "target": int(stage.get("count", 0)),
                "accepted": 0,
                "rejected_attempts": 0,
            }
            for idx, stage in enumerate(strategy)
        ]
        reject_reasons = defaultdict(int)
        
        while current_angle_idx < len(strategy):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            display = frame.copy()
            now = time.time()
            
            current_stage = strategy[current_angle_idx]
            current_angle = current_stage["angle"]
            target_for_angle = current_stage["count"]
            instruction = current_stage["instruction"]
            stage_elapsed = now - stage_start_time
            pose_relaxed = stage_elapsed >= ENROLL_POSE_RELAX_AFTER_SECONDS
            liveness_ok = not ENROLL_LIVENESS_CHALLENGE_ENABLED
            
            # ── NON-BLOCKING PAUSE UI ──────────────────────────────────
            if now < pause_until:
                overlay = display.copy()
                cv2.rectangle(overlay, (0, 0), (FRAME_WIDTH, FRAME_HEIGHT), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)
                
                cv2.putText(display, "CHANGING POSE", 
                           (FRAME_WIDTH//2 - 150, FRAME_HEIGHT//2 - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_YELLOW, 3)
                cv2.putText(display, instruction, 
                           (FRAME_WIDTH//2 - 200, FRAME_HEIGHT//2 + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2)
                cv2.putText(display, f"{int(pause_until - now) + 1}...", 
                           (FRAME_WIDTH//2 - 30, FRAME_HEIGHT//2 + 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_ORANGE, 4)
                
                cv2.imshow("Enrollment", display)
                if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                    break
                continue 
            # ───────────────────────────────────────────────────────────
            
            # Detect face based on engine
            face_detected = False
            multiple_faces = False
            quality_ok = False
            pose_ok = False
            quality_message = ""
            reject_reason = None
            yaw = None
            pitch = None

            height, width = frame.shape[:2]
            self._set_sface_input_size((width, height))
            _, faces = self.detector.detect(frame)
            face_detected = faces is not None and len(faces) > 0

            if face_detected:
                multiple_faces = len(faces) > 1
                faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                best_face = faces_sorted[0]
                x, y, w, h = best_face[0:4].astype(int)

                frame_area = float(max(1, frame.shape[0] * frame.shape[1]))
                face_area_ratio = (w * h) / frame_area

                if multiple_faces:
                    quality_message = "ONE FACE ONLY"
                    reject_reason = "multiple_faces"
                elif face_area_ratio < ENROLL_FACE_MIN_AREA_RATIO:
                    quality_message = "MOVE CLOSER"
                    reject_reason = "face_too_small"
                else:
                    roi = frame[max(0, y):max(0, y) + max(1, h), max(0, x):max(0, x) + max(1, w)]
                    if roi.size == 0:
                        quality_message = "FACE OUT OF FRAME"
                        reject_reason = "face_out_of_frame"
                    else:
                        if ENABLE_BLUR_DETECTION:
                            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                            blur_score = self.estimate_blur(roi_gray)
                            if blur_score < BLUR_THRESHOLD:
                                quality_message = "HOLD STILL (BLURRY)"
                                reject_reason = "blurry"
                            else:
                                quality_ok = True
                        else:
                            quality_ok = True

                if quality_ok:
                    yaw, pitch = self._sface_pose_signature(best_face)
                    pose_ok = self._pose_matches_stage(current_angle, yaw, pitch, relaxed=pose_relaxed)
                    if not pose_ok:
                        relax_tag = " (RELAXED)" if pose_relaxed else ""
                        quality_message = f"ADJUST TO {current_angle}{relax_tag}"
                        reject_reason = "pose_mismatch"
                else:
                    pose_ok = False
            
            # Standard UI Overlay
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (FRAME_WIDTH, 120), (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.8, display, 0.2, 0, display)
            
            cv2.putText(display, f"ANGLE: {current_angle}", 
                       (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_YELLOW, 2)
            cv2.putText(display, instruction, 
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
            cv2.putText(display, f"{angle_count}/{target_for_angle}", 
                       (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GREEN, 2)
            cv2.putText(display, f"Total: {total_collected}/{total_target} | SFACE", 
                       (FRAME_WIDTH - 350, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
            if pose_relaxed:
                cv2.putText(display, "POSE ASSIST: RELAXED", (FRAME_WIDTH - 260, 105),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_YELLOW, 2)
            if stage_elapsed >= ENROLL_STAGE_TIMEOUT_SECONDS:
                cv2.putText(display, "STAGE TIMEOUT: ADJUST LIGHT/ANGLE", (FRAME_WIDTH - 360, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_YELLOW, 2)
            
            if face_detected:
                box_color = COLOR_GREEN if (quality_ok and pose_ok) else COLOR_YELLOW
                cv2.rectangle(display, (x, y), (x+w, y+h), box_color, 3)

                if yaw is not None and pitch is not None:
                    pose_text = f"Yaw:{yaw:.2f} Pitch:{pitch:.2f}"
                    cv2.putText(display, pose_text,
                               (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)

                if quality_ok and pose_ok and ENROLL_LIVENESS_CHALLENGE_ENABLED:
                    if current_angle.upper() == "FRONT":
                        liveness_ok = True
                    else:
                        liveness_ok = self._liveness_pose_delta_ok(
                            current_angle,
                            yaw,
                            pitch,
                            front_pose_baseline
                        )
                        if not liveness_ok:
                            quality_message = "LIVENESS: TURN HEAD AS PROMPTED"
                            reject_reason = "liveness_pose_delta"
                elif quality_ok and pose_ok:
                    liveness_ok = True

                if quality_ok and pose_ok and liveness_ok:
                    stage_stable_frames += 1
                else:
                    stage_stable_frames = 0
                
                # Auto-capture logic
                if quality_ok and pose_ok and liveness_ok and stage_stable_frames >= ENROLL_MIN_STABLE_FRAMES and (now - last_capture >= capture_delay):
                    collected.append(frame.copy())
                    angle_count += 1
                    total_collected += 1
                    last_capture = now
                    stage_metrics[current_angle_idx]["accepted"] += 1

                    if current_angle.upper() == "FRONT" and yaw is not None and pitch is not None:
                        if front_pose_baseline is None:
                            front_pose_baseline = {"yaw": yaw, "pitch": pitch}
                        else:
                            # Keep a running baseline to stabilize liveness checks across front samples.
                            front_pose_baseline["yaw"] = (front_pose_baseline["yaw"] + yaw) * 0.5
                            front_pose_baseline["pitch"] = (front_pose_baseline["pitch"] + pitch) * 0.5
                    
                    if save_dir:
                        # Fix 5: Lossless PNG preserves facial landmark detail for training
                        img_path = os.path.join(save_dir, f"{current_angle}_{angle_count:02d}.png")
                        cv2.imwrite(img_path, frame)

                        if ENROLL_SAVE_FACE_CROPS:
                            crop_y1 = max(0, y)
                            crop_y2 = min(frame.shape[0], y + h)
                            crop_x1 = max(0, x)
                            crop_x2 = min(frame.shape[1], x + w)
                            face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                            if face_crop.size > 0:
                                crop_path = os.path.join(save_dir, f"{current_angle}_{angle_count:02d}_crop.png")
                                cv2.imwrite(crop_path, face_crop)
                    
                    print(f"  OK: {current_angle} [{angle_count}/{target_for_angle}]")
                    
                    if angle_count >= target_for_angle:
                        current_angle_idx += 1
                        angle_count = 0
                        stage_stable_frames = 0
                        print(f"[ENGINE] OK: {current_angle} complete!\n")
                        
                        if current_angle_idx < len(strategy):
                            pause_until = time.time() + ENROLL_STAGE_PAUSE_SECONDS
                            stage_start_time = time.time()
                elif quality_message:
                    cv2.putText(display, quality_message,
                               (FRAME_WIDTH//2 - 160, FRAME_HEIGHT - 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_YELLOW, 2)

                if now - last_capture >= capture_delay and not (quality_ok and pose_ok):
                    stage_metrics[current_angle_idx]["rejected_attempts"] += 1
                    if reject_reason:
                        reject_reasons[reject_reason] += 1
            else:
                cv2.putText(display, "⚠ NO FACE DETECTED", 
                           (FRAME_WIDTH//2 - 150, FRAME_HEIGHT - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_RED, 2)
                if now - last_capture >= capture_delay:
                    stage_metrics[current_angle_idx]["rejected_attempts"] += 1
                    reject_reasons["no_face"] += 1
            
            # Progress bar
            progress = total_collected / total_target
            bar_y = FRAME_HEIGHT - 25
            bar_w = int((FRAME_WIDTH - 40) * progress)
            cv2.rectangle(display, (20, bar_y), (FRAME_WIDTH - 20, bar_y + 15), (60, 60, 60), -1)
            cv2.rectangle(display, (20, bar_y), (20 + bar_w, bar_y + 15), COLOR_GREEN, -1)
            
            cv2.imshow("Enrollment", display)
            
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                break
        
        cap.release()
        cv2.destroyAllWindows()

        print("[ENGINE] Enrollment stage summary:")
        for stage in stage_metrics:
            print(
                f"  - {stage['angle']}: accepted {stage['accepted']}/{stage['target']} | "
                f"rejected attempts {stage['rejected_attempts']}"
            )

        if reject_reasons:
            print("[ENGINE] Rejection reason summary:")
            for reason, count in sorted(reject_reasons.items(), key=lambda item: item[1], reverse=True):
                print(f"  - {reason}: {count}")
        
        print(f"\n[ENGINE] ✓ Enrollment Complete: {len(collected)} images\n")
        return collected

    # ══════════════════════════════════════════════════════════════════════════
    # UTILITIES
    # ══════════════════════════════════════════════════════════════════════════

    def remap_confidence(self, cosine_score):
        """Fix 4: Remap SFace cosine similarity (0.0-1.0) to human-friendly percentage (85-100%).
        Match threshold (default 0.36) maps to 85%. Perfect match (1.0) maps to 100%."""
        if cosine_score < SFACE_MATCH_THRESHOLD:
            return 0.0
        
        # Linear interpolation between [threshold, 1.0] and [0.85, 1.00]
        remapped = 0.85 + (cosine_score - SFACE_MATCH_THRESHOLD) / (1.0 - SFACE_MATCH_THRESHOLD) * 0.15
        return float(max(0.0, min(1.0, remapped)))

    def update_label_map(self, label_map):
        """Update criminal name/info mapping."""
        self.label_map = label_map
        logger.info(f"[ENGINE] Label map updated: {len(label_map)} entries")

    def save_model(self):
        """Save current model to disk."""
        if self.embeddings_db:
            with open(SFACE_DB_PATH, 'wb') as f:
                pickle.dump({'embeddings': self.embeddings_db}, f)
            print(f"[ENGINE] ✓ SFace database saved")

    def estimate_blur(self, gray_image):
        """Laplacian variance blur detection."""
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        return laplacian.var()

    def _apply_clahe_normalization(self, frame):
        """Fix 6: Normalize lighting via CLAHE in LAB colour space.
        Equalises the L (luminance) channel only — colour/structure is untouched.
        ~0.3ms overhead per frame on CPU; handles dim rooms and harsh shadows."""
        try:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_ch = clahe.apply(l_ch)
            return cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
        except Exception:
            return frame  # Safe fallback — never crash the recognition loop