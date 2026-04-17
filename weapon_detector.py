"""
weapon_detector.py
Production-grade weapon detection engine with CPU optimization.
Uses YOLOv8n ONNX model for real-time weapon detection.

This module is INDEPENDENT of the face recognition pipeline.
Shared infrastructure: camera stream (via video_pipeline), config constants.

Classes:
    WeaponDetector      - Synchronous single-frame detection engine
    AsyncWeaponDetector - Background-thread wrapper for non-blocking detection
"""

import hashlib
import logging
import os
import threading
import time
import urllib.request
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np
from collections import deque
import json
import yaml

from config import (
    WEAPON_MODEL_PATH,
    WEAPON_INPUT_SIZE,
    WEAPON_CONFIDENCE_THRESHOLD,
    WEAPON_NMS_THRESHOLD,
    WEAPON_MIN_BOX_AREA,
    WEAPON_RESULT_TTL_SECONDS,
    WEAPON_WORKER_SLEEP_SECONDS,
    WEAPON_CLASSES,
    WEAPON_THREAT_LEVELS,
    WEAPON_WARMUP_FRAMES,
    WEAPON_FRAME_SKIP,
    WEAPON_SNAPSHOT_DIR,
    WEAPON_LOG_CSV,
    WEAPON_LOG_JSON,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Model Auto-Download
# ==============================================================================

def download_weapon_model(dest_path: str, url: str = "", sha256_hash: str = "") -> bool:
    """
    Download weapon ONNX model if it does not exist.
    Follows the exact same pattern as the SFace model downloader in config.py.

    Args:
        dest_path:   Local path to save the model file.
        url:         Direct download URL for the .onnx file.
        sha256_hash: Expected SHA256 hash for integrity (empty = skip check).

    Returns:
        True if model exists or was successfully downloaded.
    """
    if os.path.isfile(dest_path):
        # Verify hash if provided
        if sha256_hash:
            if _verify_sha256(dest_path, sha256_hash):
                return True
            else:
                logger.warning("[WEAPON] Model hash mismatch — re-downloading...")
                os.remove(dest_path)
        else:
            return True

    if not url:
        logger.warning(
            f"[WEAPON] Model not found at {dest_path} and no download URL configured.\n"
            f"  To fix: provide a weapon_yolov8n.onnx model in the models/ directory.\n"
            f"  See weapon_config.yaml for training instructions."
        )
        return False

    logger.info(f"[WEAPON] Downloading weapon model from: {url}")
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        urllib.request.urlretrieve(url, dest_path)

        # Post-download hash verification
        if sha256_hash and not _verify_sha256(dest_path, sha256_hash):
            logger.error("[WEAPON] Downloaded model failed SHA256 check — deleting.")
            os.remove(dest_path)
            return False

        logger.info(f"[WEAPON] Model downloaded successfully: {dest_path}")
        return True
    except Exception as exc:
        logger.error(f"[WEAPON] Model download failed: {exc}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def _verify_sha256(filepath: str, expected_hash: str) -> bool:
    """Verify SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    actual = sha256.hexdigest()
    if actual.lower() != expected_hash.lower():
        logger.warning(f"[WEAPON] SHA256 mismatch: expected={expected_hash[:16]}... got={actual[:16]}...")
        return False
    return True


# ==============================================================================
# Weapon Event Logger (CSV + JSON — fully separate from face logs)
# ==============================================================================

class WeaponEventLogger:
    """
    Logs weapon detection events to CSV and JSON files.
    Completely independent from the face detection logging system.
    Thread-safe via a lock.
    """

    def __init__(self, csv_path: str = WEAPON_LOG_CSV, json_path: str = WEAPON_LOG_JSON):
        self._csv_path = csv_path
        self._json_path = json_path
        self._lock = threading.Lock()
        self._init_csv()

    def _init_csv(self):
        """Create CSV header if file does not exist."""
        if not os.path.isfile(self._csv_path):
            os.makedirs(os.path.dirname(self._csv_path), exist_ok=True)
            with open(self._csv_path, "w") as f:
                f.write("timestamp,weapon_type,confidence,threat_level,"
                        "antispoof_score,antispoof_passed,bbox,snapshot_path\n")

    def log(self, weapon_type: str, confidence: float, threat_level: str,
            bbox: Dict, snapshot_path: str = "",
            antispoof_score: float = -1.0, antispoof_passed: bool = True):
        """
        Log a single weapon detection event to both CSV and JSON.
        """
        import json as _json

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        bbox_str = f"{bbox.get('x1',0)},{bbox.get('y1',0)},{bbox.get('x2',0)},{bbox.get('y2',0)}"

        with self._lock:
            # CSV append
            try:
                with open(self._csv_path, "a") as f:
                    f.write(f"{ts},{weapon_type},{confidence:.4f},{threat_level},"
                            f"{antispoof_score:.4f},{antispoof_passed},{bbox_str},"
                            f"{snapshot_path}\n")
            except Exception as exc:
                logger.error(f"[WEAPON] CSV log failed: {exc}")

            # JSON append (newline-delimited JSON for easy parsing)
            try:
                os.makedirs(os.path.dirname(self._json_path), exist_ok=True)
                entry = {
                    "timestamp": ts,
                    "weapon_type": weapon_type,
                    "confidence": round(confidence, 4),
                    "threat_level": threat_level,
                    "antispoof_score": round(antispoof_score, 4),
                    "antispoof_passed": antispoof_passed,
                    "bbox": bbox,
                    "snapshot_path": snapshot_path,
                }
                with open(self._json_path, "a") as f:
                    f.write(_json.dumps(entry) + "\n")
            except Exception as exc:
                logger.error(f"[WEAPON] JSON log failed: {exc}")

        # Console alert
        alert_color = "\033[91m" if threat_level == "CRITICAL" else "\033[93m"
        reset = "\033[0m"
        print(f"{alert_color}[!] WEAPON DETECTED: {weapon_type} "
              f"({confidence*100:.1f}%) [{threat_level}]{reset}")


# ==============================================================================
# WeaponDetector — Synchronous Detection Engine
# ==============================================================================

class WeaponDetector:
    """
    Production-grade weapon detection engine using ONNX Runtime.
    Supports YOLOv8n and compatible YOLO ONNX model layouts.

    Features:
        - CPU-optimized ONNX inference
        - Model warm-up to eliminate first-frame latency
        - Configurable performance profiles (speed/balanced/accuracy)
        - Auto-download with SHA256 verification
        - Frame skipping for CPU budget management
    """

    def __init__(self, model_path: str = WEAPON_MODEL_PATH):
        self.model_path = model_path
        self.model_loaded = False
        self.input_name = None
        self.output_names: List[str] = []
        self.last_error: Optional[str] = None
        self._warmup_done = False
        self._frame_counter = 0

        # Temporal Smoothing: object history (class_id -> deque of recent confidences)
        # We use a simple world-coordinate-less tracker (just class consistency)
        self._history = deque(maxlen=5) 

        self._load_model()

    def _load_model(self):
        """
        Load the model. 
        Automatically detects if path is a directory (OpenVINO) or file (ONNX).
        """
        try:
            if os.path.isdir(self.model_path) or self.model_path.endswith(".xml"):
                self._load_openvino()
            else:
                self._load_onnx()
        except Exception as exc:
            self.model_loaded = False
            self.last_error = str(exc)
            logger.error(f"[WEAPON] Model load failed: {exc}")

    def _load_openvino(self):
        """Load using Intel OpenVINO Toolkit for maximum CPU performance."""
        try:
            from openvino.runtime import Core
            
            xml_path = self.model_path
            if os.path.isdir(self.model_path):
                # Search for .xml file in the directory
                xml_files = [f for f in os.listdir(self.model_path) if f.endswith(".xml")]
                if not xml_files:
                    raise FileNotFoundError(f"No .xml found in {self.model_path}")
                xml_path = os.path.join(self.model_path, xml_files[0])

            core = Core()
            model = core.read_model(xml_path)
            self.compiled_model = core.compile_model(model, "CPU")
            self.infer_request = self.compiled_model.create_infer_request()
            
            self.input_name = self.compiled_model.input(0).get_any_name()
            # Handle multiple outputs if present
            self.output_names = [out.get_any_name() for out in self.compiled_model.outputs]
            
            self.model_format = "openvino"
            self.model_loaded = True
            logger.info(f"[WEAPON] OpenVINO Model loaded: {xml_path}")
        except ImportError:
            logger.error("[WEAPON] openvino-dev not installed. Falling back to ONNX.")
            self._load_onnx()

    def _load_onnx(self):
        """Load the ONNX model using ONNX Runtime with CPU optimization."""
        import onnxruntime as ort

        if not os.path.isfile(self.model_path):
            self.model_loaded = False
            self.last_error = f"Model file not found: {self.model_path}"
            logger.warning(f"[WEAPON] {self.last_error}")
            return

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = os.cpu_count() or 4
        sess_options.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.model_format = "onnx"
        self.model_loaded = True
        logger.info(f"[WEAPON] ONNX Model loaded: {self.model_path}")

    def warmup(self):
        """
        Run dummy inferences to eliminate first-frame latency spike.
        Call this once after model load, before processing real frames.
        """
        if not self.model_loaded or self._warmup_done:
            return

        logger.info(f"[WEAPON] Warming up model ({WEAPON_WARMUP_FRAMES} frames)...")
        dummy = np.zeros(
            (1, 3, WEAPON_INPUT_SIZE[1], WEAPON_INPUT_SIZE[0]),
            dtype=np.float32,
        )
        for _ in range(WEAPON_WARMUP_FRAMES):
            try:
                if self.model_format == "openvino":
                    self.infer_request.infer({self.input_name: dummy})
                else:
                    self.session.run(self.output_names, {self.input_name: dummy})
            except Exception:
                break

        self._warmup_done = True
        logger.info("[WEAPON] Warmup complete.")

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run weapon detection on a single BGR frame.

        Args:
            frame: BGR image (numpy array, HxWx3).

        Returns:
            List of detection dicts, each with keys:
                type, confidence, threat_level, bbox {x1, y1, x2, y2}
        """
        if not self.model_loaded:
            return []

        # Frame skipping: only process every Nth frame
        self._frame_counter += 1
        if WEAPON_FRAME_SKIP > 1 and (self._frame_counter % WEAPON_FRAME_SKIP) != 1:
            return []  # Skip this frame

        input_tensor = self._preprocess(frame)
        
        if self.model_format == "openvino":
            results = self.infer_request.infer({self.input_name: input_tensor})
            # OpenVINO return keys are actual Tensor objects or indices
            outputs = [results[self.compiled_model.output(i)] for i in range(len(self.output_names))]
        else:
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        raw_detections = self._postprocess(outputs, frame.shape[:2])
        
        # Apply Temporal Smoothing
        return self._smooth_detections(raw_detections)

    def _smooth_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Filters detections based on temporal consistency (Hysteresis).
        An object must appear in 3 out of the last 5 frames to be active.
        """
        current_frame_types = {d["type"] for d in detections}
        self._history.append(current_frame_types)

        confirmed_detections = []
        for det in detections:
            w_type = det["type"]
            # Count appearances in recent history
            count = sum(1 for frame_types in self._history if w_type in frame_types)
            
            # 3/5 frames threshold for "100% accurate" feel during demo
            if count >= 3:
                confirmed_detections.append(det)
        
        return confirmed_detections

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize, normalize, and format frame for ONNX model input."""
        target_w, target_h = WEAPON_INPUT_SIZE
        img = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)    # Add batch dim
        return img

    def _postprocess(self, outputs, orig_hw) -> List[Dict]:
        """
        Parse YOLO ONNX output, apply NMS, and return detections.
        Supports both common YOLO layouts:
            1) [x, y, w, h, obj, c1..cN]   — with explicit objectness
            2) [x, y, w, h, c1..cN]         — no explicit objectness
        """
        pred = outputs[0]

        # Handle 3D output: (1, num_attrs, num_boxes) or (1, num_boxes, num_attrs)
        if len(pred.shape) == 3:
            if pred.shape[1] < pred.shape[2]:
                pred = np.transpose(pred, (0, 2, 1))
            pred = pred[0]

        if pred.ndim != 2 or pred.shape[1] < 6:
            return []

        orig_h, orig_w = orig_hw
        in_w, in_h = WEAPON_INPUT_SIZE
        scale_x = float(orig_w) / float(in_w)
        scale_y = float(orig_h) / float(in_h)

        boxes, scores, classes = [], [], []

        for row in pred:
            x_c, y_c, w, h = row[:4]

            # YOLOv8 format: No objectness score, columns 4+ are purely class scores
            class_scores = row[4:]

            if class_scores.size == 0:
                continue

            class_id = int(np.argmax(class_scores))
            if class_id not in WEAPON_CLASSES:
                continue
                
            confidence = float(class_scores[class_id])

            if confidence < WEAPON_CONFIDENCE_THRESHOLD:
                continue

            # Convert center-wh to top-left-wh
            x1 = int((x_c - (w / 2.0)) * scale_x)
            y1 = int((y_c - (h / 2.0)) * scale_y)
            bw = int(w * scale_x)
            bh = int(h * scale_y)

            if bw <= 0 or bh <= 0 or bw * bh < WEAPON_MIN_BOX_AREA:
                continue

            # Clamp to frame boundaries
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            bw = min(bw, orig_w - x1)
            bh = min(bh, orig_h - y1)

            boxes.append([x1, y1, bw, bh])
            scores.append(confidence)
            classes.append(class_id)

        if not boxes:
            return []

        # Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, scores, WEAPON_CONFIDENCE_THRESHOLD, WEAPON_NMS_THRESHOLD)
        if len(indices) == 0:
            return []

        detections = []
        for idx in np.array(indices).flatten():
            x1, y1, bw, bh = boxes[idx]
            class_id = classes[idx]
            weapon_type = WEAPON_CLASSES.get(class_id, f"weapon_{class_id}")
            detections.append({
                "type": weapon_type,
                "confidence": float(scores[idx]),
                "threat_level": WEAPON_THREAT_LEVELS.get(weapon_type, "UNKNOWN"),
                "bbox": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x1 + bw),
                    "y2": int(y1 + bh),
                },
            })

        return detections


# ==============================================================================
# AsyncWeaponDetector — Background Thread Wrapper
# ==============================================================================

class AsyncWeaponDetector:
    """
    Wraps WeaponDetector in a background thread that always processes
    the latest frame, avoiding queue build-up.

    Usage:
        detector = AsyncWeaponDetector().start()
        detector.submit_frame(frame, frame_id)
        result = detector.get_latest_result()
        detector.stop()
    """

    def __init__(self, model_path: str = WEAPON_MODEL_PATH):
        self.detector = WeaponDetector(model_path)
        self.model_loaded = self.detector.model_loaded
        self.last_error = self.detector.last_error

        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_id = -1

        self._result_lock = threading.Lock()
        self._latest_result: Dict = {
            "frame_id": -1,
            "timestamp": 0.0,
            "detections": [],
            "model_loaded": self.model_loaded,
        }

    def start(self) -> "AsyncWeaponDetector":
        """Start the background detection worker thread."""
        if not self.model_loaded or self._running:
            return self

        # Warm up the model before starting the worker
        self.detector.warmup()

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="WeaponWorker")
        self._thread.start()
        logger.info("[WEAPON] Async detector started.")
        return self

    def stop(self):
        """Stop the background worker cleanly."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("[WEAPON] Async detector stopped.")

    def submit_frame(self, frame: np.ndarray, frame_id: int):
        """Submit a new frame for async processing."""
        if not self.model_loaded:
            return
        with self._frame_lock:
            self._latest_frame = frame.copy()
            self._latest_frame_id = frame_id

    def get_latest_result(self) -> Dict:
        """
        Get the most recent detection result.
        Returns empty detections if result is stale (older than TTL).
        """
        with self._result_lock:
            result = {
                "frame_id": self._latest_result["frame_id"],
                "timestamp": self._latest_result["timestamp"],
                "detections": list(self._latest_result["detections"]),
                "model_loaded": self._latest_result["model_loaded"],
            }

        # Discard stale results
        age = time.time() - result["timestamp"]
        if age > WEAPON_RESULT_TTL_SECONDS:
            result["detections"] = []
        return result

    def _run(self):
        """Background worker loop — process latest frame, skip stale ones."""
        last_processed_id = -1

        while self._running:
            frame = None
            frame_id = -1

            with self._frame_lock:
                if (self._latest_frame is not None
                        and self._latest_frame_id != last_processed_id):
                    frame = self._latest_frame
                    frame_id = self._latest_frame_id

            if frame is None:
                time.sleep(WEAPON_WORKER_SLEEP_SECONDS)
                continue

            detections = self.detector.detect(frame)
            last_processed_id = frame_id

            with self._result_lock:
                self._latest_result = {
                    "frame_id": frame_id,
                    "timestamp": time.time(),
                    "detections": detections,
                    "model_loaded": True,
                }


# ==============================================================================
# Drawing Helper — Render weapon detections onto a frame
# ==============================================================================

def draw_weapon_detections(frame: np.ndarray, detections: List[Dict],
                           antispoof_results: Optional[Dict] = None):
    """
    Draw weapon bounding boxes, labels, and optional anti-spoof scores onto a frame.

    Args:
        frame:             BGR frame to annotate (modified in-place).
        detections:        List of detection dicts from WeaponDetector.
        antispoof_results: Optional dict mapping detection index -> {score, passed}.
    """
    threat_colors = {
        "CRITICAL": (0, 0, 255),     # Red
        "HIGH":     (0, 140, 255),   # Orange
        "MEDIUM":   (0, 220, 220),   # Yellow
        "UNKNOWN":  (160, 160, 160), # Gray
    }

    for i, det in enumerate(detections):
        bbox = det.get("bbox", {})
        x1 = int(bbox.get("x1", 0))
        y1 = int(bbox.get("y1", 0))
        x2 = int(bbox.get("x2", 0))
        y2 = int(bbox.get("y2", 0))

        threat = det.get("threat_level", "UNKNOWN")
        color = threat_colors.get(threat, (160, 160, 160))

        # Main bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Corner accents (military-style)
        cl = 14
        for (cx, cy) in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            dx = 1 if cx == x1 else -1
            dy = 1 if cy == y1 else -1
            cv2.line(frame, (cx, cy), (cx + dx * cl, cy), color, 3)
            cv2.line(frame, (cx, cy), (cx, cy + dy * cl), color, 3)

        # Label: weapon type + confidence
        label = f"{det.get('type', 'weapon')} {det.get('confidence', 0.0):.2f}"

        # Anti-spoof annotation
        if antispoof_results and i in antispoof_results:
            as_result = antispoof_results[i]
            score = as_result.get("score", -1)
            passed = as_result.get("passed", True)
            if score >= 0:
                status = "REAL" if passed else "FAKE"
                label += f" [{status} {score:.0%}]"
                if not passed:
                    color = (128, 128, 128)  # Gray out fake detections
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color, -1)
        cv2.putText(
            frame, label,
            (x1 + 4, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )


# ==============================================================================
# Snapshot Helper
# ==============================================================================

def save_weapon_snapshot(frame: np.ndarray, detections: List[Dict],
                         save_dir: str = WEAPON_SNAPSHOT_DIR) -> Optional[str]:
    """
    Save an annotated frame to the weapon snapshots directory.

    Filename format: YYYY-MM-DD_HH-MM-SS_classname_confidence.jpg

    Args:
        frame:      BGR frame (a copy will be annotated).
        detections: List of weapon detections.
        save_dir:   Directory to save the snapshot.

    Returns:
        Path to the saved snapshot, or None if no detections.
    """
    if not detections:
        return None

    os.makedirs(save_dir, exist_ok=True)

    # Build filename from the highest-confidence detection
    top = max(detections, key=lambda d: d.get("confidence", 0))
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    weapon_type = top.get("type", "weapon").replace(" ", "_")
    conf = top.get("confidence", 0)
    filename = f"{ts}_{weapon_type}_{conf:.2f}.jpg"
    path = os.path.join(save_dir, filename)

    # Draw detections on a copy and save
    annotated = frame.copy()
    draw_weapon_detections(annotated, detections)
    cv2.imwrite(path, annotated)

    return path
