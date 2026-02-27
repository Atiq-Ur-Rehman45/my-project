"""
==============================================================
  AI Face Recognition System — Central Configuration
==============================================================
"""

import os

# ── Base Paths ────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DATA_DIR         = os.path.join(BASE_DIR, "data")
CRIMINAL_DB_DIR  = os.path.join(DATA_DIR, "criminal_db")
CAPTURED_DIR     = os.path.join(DATA_DIR, "captured_faces")
TRAINING_DIR     = os.path.join(DATA_DIR, "training_data")
MODELS_DIR       = os.path.join(BASE_DIR, "models")
LOGS_DIR         = os.path.join(BASE_DIR, "logs")

# ── Database ──────────────────────────────────────────────────────────────────
DB_PATH          = os.path.join(DATA_DIR, "criminal_records.db")

# ── Model Paths ───────────────────────────────────────────────────────────────
LBPH_MODEL_PATH  = os.path.join(MODELS_DIR, "lbph_face_model.xml")

# ── Cascade Classifier ────────────────────────────────────────────────────────
import cv2
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH  = cv2.data.haarcascades + "haarcascade_eye.xml"

# ── Face Detection Parameters ─────────────────────────────────────────────────
DETECTION_SCALE_FACTOR   = 1.1       # Haar cascade scale factor
DETECTION_MIN_NEIGHBORS  = 5         # Higher = fewer false positives
DETECTION_MIN_SIZE       = (60, 60)  # Minimum face size in pixels

# ── Recognition Thresholds ────────────────────────────────────────────────────
# LBPH distance: lower = more similar. Values above threshold → "Unknown"
RECOGNITION_CONFIDENCE_THRESHOLD = 70   # Tune: 50 (strict) → 85 (lenient)
UNKNOWN_LABEL                     = "Unknown"

# ── Image Capture Settings ────────────────────────────────────────────────────
ENROLL_FRAME_COUNT    = 60    # How many face images to capture per person
ENROLL_CAPTURE_DELAY  = 0.05  # Seconds between captures (to get variety)
FRAME_WIDTH           = 640
FRAME_HEIGHT          = 480
CAMERA_INDEX          = 0     # 0 = default webcam

# ── UI Colors (BGR format for OpenCV) ─────────────────────────────────────────
COLOR_GREEN    = (0,   220,  0  )
COLOR_RED      = (0,   0,   220 )
COLOR_YELLOW   = (0,   200, 200 )
COLOR_BLUE     = (220, 100,  0  )
COLOR_WHITE    = (255, 255, 255 )
COLOR_BLACK    = (0,   0,   0   )
COLOR_ORANGE   = (0,   165, 255 )

# ── Alert Settings ────────────────────────────────────────────────────────────
ALERT_COOLDOWN_SECONDS = 30   # Min seconds between repeated alerts for same person
SNAPSHOT_ON_DETECTION  = True # Save snapshot when criminal is recognized

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = os.path.join(LOGS_DIR, "system.log")

# ── Ensure all directories exist ──────────────────────────────────────────────
for _dir in [DATA_DIR, CRIMINAL_DB_DIR, CAPTURED_DIR, TRAINING_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(_dir, exist_ok=True)
