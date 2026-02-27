"""
==============================================================
  AI Face Recognition System — Face Recognition Engine
  Uses: OpenCV Haar Cascade (detection) + LBPH (recognition)
==============================================================
"""

import cv2
import numpy as np
import os
import time
import logging
from config import (
    FACE_CASCADE_PATH, EYE_CASCADE_PATH, LBPH_MODEL_PATH,
    DETECTION_SCALE_FACTOR, DETECTION_MIN_NEIGHBORS, DETECTION_MIN_SIZE,
    RECOGNITION_CONFIDENCE_THRESHOLD, UNKNOWN_LABEL
)

logger = logging.getLogger(__name__)


class FaceEngine:
    """
    Core face detection and recognition engine.
    
    Detection  → Haar Cascade (fast, runs on CPU, no GPU needed)
    Recognition → LBPH (Local Binary Pattern Histogram)
                  - Works well with limited training images
                  - Lightweight and interpretable confidence scores
    """

    def __init__(self):
        # ── Load detectors ────────────────────────────────────────────────────
        self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        self.eye_cascade  = cv2.CascadeClassifier(EYE_CASCADE_PATH)

        if self.face_cascade.empty():
            raise RuntimeError(f"[ENGINE] Failed to load face cascade from: {FACE_CASCADE_PATH}")

        # ── LBPH Recognizer ───────────────────────────────────────────────────
        self.recognizer       = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=float(RECOGNITION_CONFIDENCE_THRESHOLD)
        )
        self.model_loaded     = False
        self.label_map        = {}   # {int_label: criminal_dict}
        self.confidence_thresh = RECOGNITION_CONFIDENCE_THRESHOLD

        # Try loading existing model
        self._load_model()
        logger.info("[ENGINE] FaceEngine initialized.")

    # ── Model Persistence ─────────────────────────────────────────────────────

    def _load_model(self):
        """Load a pre-trained LBPH model if it exists."""
        if os.path.exists(LBPH_MODEL_PATH):
            try:
                self.recognizer.read(LBPH_MODEL_PATH)
                self.model_loaded = True
                logger.info(f"[ENGINE] Model loaded from {LBPH_MODEL_PATH}")
                print(f"[ENGINE] ✓ Trained model loaded from disk.")
            except Exception as e:
                logger.warning(f"[ENGINE] Could not load model: {e}")
                self.model_loaded = False
        else:
            print("[ENGINE] No trained model found. Please enroll faces and train first.")

    def save_model(self):
        """Persist the trained LBPH model to disk."""
        self.recognizer.save(LBPH_MODEL_PATH)
        print(f"[ENGINE] ✓ Model saved → {LBPH_MODEL_PATH}")
        logger.info(f"[ENGINE] Model saved to {LBPH_MODEL_PATH}")

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect_faces(self, frame, equalize=True):
        """
        Detect faces in a BGR frame.
        
        Returns list of (x, y, w, h) tuples.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if equalize:
            gray = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=DETECTION_SCALE_FACTOR,
            minNeighbors=DETECTION_MIN_NEIGHBORS,
            minSize=DETECTION_MIN_SIZE,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return list(faces) if len(faces) > 0 else [], gray

    def preprocess_face(self, gray, x, y, w, h, target_size=(100, 100)):
        """
        Extract and preprocess a face ROI for recognition.
        - Crop with padding
        - Resize to fixed size
        - CLAHE for illumination normalization
        """
        # Add small padding without going out of bounds
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(gray.shape[1], x + w + pad)
        y2 = min(gray.shape[0], y + h + pad)

        face_roi = gray[y1:y2, x1:x2]

        # CLAHE — enhances local contrast, handles lighting variation
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        face_roi = clahe.apply(face_roi)

        # Resize to standard size
        face_roi = cv2.resize(face_roi, target_size)
        return face_roi

    # ── Recognition ───────────────────────────────────────────────────────────

    def recognize_face(self, gray, x, y, w, h):
        """
        Recognize a detected face.
        
        Returns (label_int, confidence, name_str)
        - confidence is the LBPH distance (lower = more similar)
        - If above threshold → UNKNOWN
        """
        if not self.model_loaded:
            return -1, 999.0, UNKNOWN_LABEL

        face_roi = self.preprocess_face(gray, x, y, w, h)

        try:
            label, confidence = self.recognizer.predict(face_roi)
        except cv2.error as e:
            logger.warning(f"[ENGINE] Predict error: {e}")
            return -1, 999.0, UNKNOWN_LABEL

        if confidence > self.confidence_thresh:
            return -1, confidence, UNKNOWN_LABEL

        # Lookup name from label map
        criminal = self.label_map.get(label)
        name = criminal["name"] if criminal else f"Label_{label}"
        return label, confidence, name

    def recognize_all_faces(self, frame):
        """
        Detect all faces in frame and recognize each one.
        
        Returns:
            results: list of {x, y, w, h, label, confidence, name, criminal}
            gray:    grayscale frame
        """
        faces, gray = self.detect_faces(frame)
        results = []

        for (x, y, w, h) in faces:
            label, confidence, name = self.recognize_face(gray, x, y, w, h)
            criminal = self.label_map.get(label) if label != -1 else None
            results.append({
                "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                "label": label,
                "confidence": confidence,
                "name": name,
                "criminal": criminal,
                "is_known": name != UNKNOWN_LABEL
            })

        return results, gray

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, training_data):
        """
        Train (or re-train) the LBPH model.
        
        Args:
            training_data: list of (face_image_grayscale_100x100, int_label)
        """
        if not training_data:
            print("[ENGINE] No training data provided.")
            return False

        images = [item[0] for item in training_data]
        labels = np.array([item[1] for item in training_data], dtype=np.int32)

        print(f"[ENGINE] Training on {len(images)} face samples across {len(set(labels))} person(s)...")
        self.recognizer.train(images, labels)
        self.model_loaded = True
        self.save_model()
        print(f"[ENGINE] ✓ Training complete!")
        return True

    def update_training(self, new_data):
        """
        Incrementally update the model with new training samples.
        Only works after initial training.
        """
        if not self.model_loaded:
            return self.train(new_data)

        images = [item[0] for item in new_data]
        labels = np.array([item[1] for item in new_data], dtype=np.int32)
        self.recognizer.update(images, labels)
        self.save_model()
        print(f"[ENGINE] ✓ Model updated with {len(images)} new samples.")
        return True

    def update_label_map(self, label_map):
        """Update the internal label → criminal dict mapping."""
        self.label_map = label_map
        logger.info(f"[ENGINE] Label map updated with {len(label_map)} entries.")

    # ── Enrollment Helper ─────────────────────────────────────────────────────

    def collect_face_samples(self, camera_index=0, target_count=60, label=0, save_dir=None):
        """
        Open webcam and collect face samples for a person.
        
        Returns list of (processed_face_array, label) tuples.
        Also saves images to save_dir if provided.
        """
        from config import (
            FRAME_WIDTH, FRAME_HEIGHT, ENROLL_CAPTURE_DELAY,
            COLOR_GREEN, COLOR_YELLOW, COLOR_WHITE, COLOR_ORANGE
        )

        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        if not cap.isOpened():
            raise RuntimeError(f"[ENGINE] Cannot open camera index {camera_index}")

        collected = []
        count = 0
        last_capture = 0

        print(f"\n[ENGINE] Webcam opened. Collecting {target_count} face samples...")
        print("[ENGINE] → Position your face in the frame.")
        print("[ENGINE] → Press 'Q' to cancel early.\n")

        while count < target_count:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Mirror effect
            display = frame.copy()

            faces, gray = self.detect_faces(frame)
            now = time.time()

            for (x, y, w, h) in faces:
                cv2.rectangle(display, (x, y), (x + w, y + h), COLOR_GREEN, 2)

                # Capture at interval to get varied poses
                if now - last_capture >= ENROLL_CAPTURE_DELAY and count < target_count:
                    face_img = self.preprocess_face(gray, x, y, w, h)
                    collected.append((face_img, label))
                    count += 1
                    last_capture = now

                    # Save to disk if directory given
                    if save_dir:
                        img_path = os.path.join(save_dir, f"face_{count:04d}.jpg")
                        cv2.imwrite(img_path, face_img)

            # ── HUD Overlay ───────────────────────────────────────────────────
            progress = count / target_count
            bar_w    = int(FRAME_WIDTH * 0.6 * progress)
            cv2.rectangle(display, (20, FRAME_HEIGHT - 50), (20 + int(FRAME_WIDTH * 0.6), FRAME_HEIGHT - 25), (50, 50, 50), -1)
            cv2.rectangle(display, (20, FRAME_HEIGHT - 50), (20 + bar_w, FRAME_HEIGHT - 25), COLOR_GREEN, -1)

            # Status text
            status = f"Captured: {count}/{target_count}"
            cv2.putText(display, status, (20, FRAME_HEIGHT - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)

            instruction = "Look forward, then turn slightly left/right"
            cv2.putText(display, instruction, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_YELLOW, 1)

            if not faces:
                cv2.putText(display, "No face detected — move closer", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_ORANGE, 1)

            cv2.imshow("Face Enrollment — Press Q to cancel", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("[ENGINE] Enrollment cancelled by user.")
                break

        cap.release()
        cv2.destroyAllWindows()

        print(f"[ENGINE] Collected {len(collected)} face samples.")
        return collected
