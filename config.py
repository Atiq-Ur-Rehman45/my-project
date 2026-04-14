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

# ══════════════════════════════════════════════════════════════════════════════
# SFACE (Deep Learning) Configuration
# ══════════════════════════════════════════════════════════════════════════════

# ── Model Paths ───────────────────────────────────────────────────────────────
YUNET_MODEL_PATH = os.path.join(MODELS_DIR, "yunet.onnx")
SFACE_MODEL_PATH = os.path.join(MODELS_DIR, "sface.onnx")
SFACE_DB_PATH    = os.path.join(MODELS_DIR, "sface_embeddings.pkl")

# ── YuNet Detection Parameters ────────────────────────────────────────────────
YUNET_INPUT_SIZE = (320, 320)       # Input size for YuNet (don't change)
YUNET_SCORE_THRESHOLD = 0.6         # Detection confidence (0.0-1.0)
                                     # 0.5 = more detections, 0.7 = fewer false positives
YUNET_NMS_THRESHOLD = 0.3           # Non-maximum suppression (0.0-1.0)
YUNET_TOP_K = 5000                  # Max detections to consider

# ── SFace Recognition Parameters ──────────────────────────────────────────────
SFACE_OPERATING_PROFILE = "BALANCED"   # BALANCED, SECURITY_FIRST, RECALL_FIRST

if SFACE_OPERATING_PROFILE == "SECURITY_FIRST":
    SFACE_MATCH_THRESHOLD_ACQUIRE = 0.60 # Very strict, lowest false positives
    SFACE_MATCH_THRESHOLD_MAINTAIN = 0.57
    SFACE_MARGIN_THRESHOLD_ACQUIRE = 0.08
    SFACE_MARGIN_THRESHOLD_MAINTAIN = 0.06
    SFACE_CONSENSUS_FRAMES = 4           # Need more agreeing frames before confirmed known
elif SFACE_OPERATING_PROFILE == "RECALL_FIRST":
    SFACE_MATCH_THRESHOLD_ACQUIRE = 0.50 # Lenient, higher recall
    SFACE_MATCH_THRESHOLD_MAINTAIN = 0.47
    SFACE_MARGIN_THRESHOLD_ACQUIRE = 0.04
    SFACE_MARGIN_THRESHOLD_MAINTAIN = 0.03
    SFACE_CONSENSUS_FRAMES = 2
else:
    SFACE_MATCH_THRESHOLD_ACQUIRE = 0.50 # Tuned: relaxed for distance; margin gate + consensus prevent false positives
    SFACE_MATCH_THRESHOLD_MAINTAIN = 0.47
    SFACE_MARGIN_THRESHOLD_ACQUIRE = 0.045
    SFACE_MARGIN_THRESHOLD_MAINTAIN = 0.03
    SFACE_CONSENSUS_FRAMES = 2

SFACE_MATCH_THRESHOLD = SFACE_MATCH_THRESHOLD_ACQUIRE  # Compatibility alias
SFACE_MARGIN_THRESHOLD = SFACE_MARGIN_THRESHOLD_ACQUIRE
SFACE_CONSENSUS_WINDOW = 6              # Track this many recent frames per face track
SFACE_KNOWN_HOLD_FRAMES = 12            # Keep confirmed identity for N frames (~1s) during blur/motion
SFACE_RECOG_MIN_FACE_AREA_RATIO = 0.015 # Tuned: allows recognition up to ~2.5m (was 0.035 = ~1m limit)
SFACE_RECOG_BLUR_THRESHOLD = 40.0       # Strict threshold for initial identification
SFACE_RECOG_BLUR_THRESHOLD_MAINTAIN = 20.0 # Lenient threshold for walking/moving subjects

# Multi-face stabilization cache
SFACE_CACHE_REUSE_FRAMES = 2        # Reuse last recognition for up to N frames if same face box persists
SFACE_CACHE_TTL_SECONDS = 0.7       # Drop stale cached tracks quickly to avoid ghost identities
SFACE_CACHE_IOU_THRESHOLD = 0.65    # Strict overlap needed to reuse a cached identity
SFACE_CACHE_MAX_TRACKS = 64         # Hard cap to keep cache operations bounded
SFACE_SKIP_RECOGNITION_FRAMES = 2   # Skip expensive DNN inference for cached tracks within N frames (FPS boost)

# ── Tier 2 Accuracy Bundle (Advanced Software Optimization) ──────────────────
SFACE_OUTLIER_THRESHOLD    = 0.40  # Max avg cosine distance between enrollment embeddings
SFACE_OUTLIER_MIN_SAMPLES  = 10    # Never prune a person below this many samples
SFACE_SHARPEN_MIN_SIZE     = 80    # Apply USM if width/height is below this (px)
ENABLE_USM_DEBUG           = True  # Draw indicator when sharpening is active

# ── Elite-Level Accuracy Refinements ─────────────────────────────────────────
SFACE_TOP_K_MATCH          = 3     # Average the top-N matches per person for robustness
SFACE_MATCH_FLOOR          = 0.35  # Hard floor for identification (ignore lower scores)

# ── Common Settings ───────────────────────────────────────────────────────────
UNKNOWN_LABEL = "Unknown"

# ══════════════════════════════════════════════════════════════════════════════
# Camera & Enrollment Settings
# ══════════════════════════════════════════════════════════════════════════════

# ── Camera Configuration ──────────────────────────────────────────────────────
FRAME_WIDTH           = 640         # Optimized for CPU Deep Learning
FRAME_HEIGHT          = 480         # Drastically increases FPS
CAMERA_INDEX          = 0           # 0 = default webcam, 1 = external

# Camera optimization flags
CAMERA_FPS_TARGET     = 30          # Request 30 FPS from camera
CAMERA_AUTOFOCUS      = False       # Disable autofocus for stability
CAMERA_BUFFER_SIZE    = 1           # Reduce lag (1-3)
WARMUP_FRAMES         = 10          # Discard first N frames (camera settling)
ASYNC_CAMERA_CAPTURE  = True        # Read camera frames on a background thread

# ── Enrollment Settings ───────────────────────────────────────────────────────
ENROLL_FRAME_COUNT = 20         # SFace: 20 images improves identity separation

ENROLL_CAPTURE_DELAY  = 0.4         # Seconds between auto-captures
ENROLL_COUNTDOWN      = 3           # Countdown before enrollment starts
ENROLL_STAGE_PAUSE_SECONDS = 2.5    # Pause between angle stages
ENROLL_SECURITY_PROFILE = "STRICT"  # STRICT or BALANCED

# Enrollment quality/pose assistance
ENROLL_FACE_MIN_AREA_RATIO = 0.04   # Minimum face area ratio in frame for reliable enrollment
ENROLL_POSE_RELAX_AFTER_SECONDS = 8.0  # After this, pose thresholds are relaxed to reduce user friction
ENROLL_SAVE_FACE_CROPS = True       # Save aligned crop-like face ROI alongside full frame for quick QA
ENROLL_STAGE_TIMEOUT_SECONDS = 20.0  # Warn operator if one stage takes too long

# Basic anti-spoof/liveness controls during enrollment
ENROLL_LIVENESS_CHALLENGE_ENABLED = True   # Require pose transition from front baseline
ENROLL_REQUIRED_POSE_DELTA = 0.06          # Minimum yaw/pitch shift to pass movement challenge
ENROLL_MIN_STABLE_FRAMES = 3               # Require N stable valid frames before each capture

# Balanced profile softens strict defaults without changing code paths
if ENROLL_SECURITY_PROFILE == "BALANCED":
    ENROLL_REQUIRED_POSE_DELTA = 0.04
    ENROLL_MIN_STABLE_FRAMES = 2
    ENROLL_FACE_MIN_AREA_RATIO = 0.035

# ── Multi-Angle Enrollment Strategy ───────────────────────────────────────────
# SFace: Balanced quality and diversity for stronger embeddings
ENROLLMENT_STRATEGY = [
    {"angle": "FRONT", "count": 8, "instruction": "Look STRAIGHT at camera"},
    {"angle": "LEFT",  "count": 4, "instruction": "Turn head to the LEFT"},
    {"angle": "RIGHT", "count": 4, "instruction": "Turn head to the RIGHT"},
    {"angle": "UP",    "count": 2, "instruction": "Tilt head UP slightly"},
    {"angle": "DOWN",  "count": 2, "instruction": "Tilt head DOWN slightly"},
]

# ── Image Quality Settings ────────────────────────────────────────────────────
ENABLE_BLUR_DETECTION = True        # Reject blurry frames during enrollment
BLUR_THRESHOLD        = 80.0        # Laplacian variance threshold

# ══════════════════════════════════════════════════════════════════════════════
# UI & Display Settings
# ══════════════════════════════════════════════════════════════════════════════

# ── UI Colors (BGR format for OpenCV) ─────────────────────────────────────────
COLOR_GREEN    = (0,   220,  0  )
COLOR_RED      = (0,   0,   220 )
COLOR_YELLOW   = (0,   200, 200 )
COLOR_BLUE     = (220, 100,  0  )
COLOR_WHITE    = (255, 255, 255 )
COLOR_BLACK    = (0,   0,   0   )
COLOR_ORANGE   = (0,   165, 255 )
COLOR_PURPLE   = (255, 0,   255 )

# ── Alert Settings ────────────────────────────────────────────────────────────
ALERT_COOLDOWN_SECONDS = 30         # Min seconds between repeated alerts
SNAPSHOT_ON_DETECTION  = True       # Save snapshot when criminal detected
ASYNC_ALERT_PROCESSING = True       # Offload alert I/O away from the live loop
ALERT_WORKER_QUEUE_SIZE = 32        # Bounded queue to avoid runaway alert lag

# ── Weapon Detection (Async Worker) ──────────────────────────────────────────
ENABLE_WEAPON_DETECTION = True
WEAPON_MODEL_PATH = os.path.join(MODELS_DIR, "weapon_yolov8n.onnx")
WEAPON_INPUT_SIZE = (640, 640)
WEAPON_CONFIDENCE_THRESHOLD = 0.35
WEAPON_NMS_THRESHOLD = 0.45
WEAPON_MIN_BOX_AREA = 300
WEAPON_RESULT_TTL_SECONDS = 0.35      # Drop stale async detections quickly
WEAPON_WORKER_SLEEP_SECONDS = 0.001   # Small throttle to avoid busy spin
WEAPON_ALERT_COOLDOWN_SECONDS = 15
WEAPON_ALERT_ON_UNKNOWN = True
WEAPON_SNAPSHOT_ON_DETECTION = True

# Model class IDs expected from the weapon model output
WEAPON_CLASSES = {
    0: "handgun",
    1: "rifle",
    2: "knife",
    3: "scissors",
}

WEAPON_THREAT_LEVELS = {
    "handgun": "CRITICAL",
    "rifle": "CRITICAL",
    "knife": "HIGH",
    "scissors": "MEDIUM",
}

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = os.path.join(LOGS_DIR, "system.log")

# ══════════════════════════════════════════════════════════════════════════════
# Web Server Settings
# ══════════════════════════════════════════════════════════════════════════════
WEB_HOST          = "0.0.0.0"
WEB_PORT          = 5000
WEB_DEBUG         = False                   # Set True during development
WEB_SECRET_KEY    = "fyp-face-recog-2026"

# ── MJPEG Video Streaming ─────────────────────────────────────────────────────
MJPEG_QUALITY     = 75                      # JPEG encode quality (1-100)
MJPEG_MAX_FPS     = 25                      # Cap stream FPS to spare CPU

# ── Uploaded video files (drag-and-drop) ─────────────────────────────────────
UPLOAD_DIR        = os.path.join(DATA_DIR, "uploads")
UPLOAD_MAX_MB     = 500                     # Max upload size in MB
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}

# ══════════════════════════════════════════════════════════════════════════════
# Auto-Download Models (SFace Only)
# ══════════════════════════════════════════════════════════════════════════════

def download_models_if_needed():
    """Auto-download YuNet and SFace models if not present."""
    import urllib.request
    models_to_download = []
    
    if not os.path.exists(YUNET_MODEL_PATH):
        models_to_download.append({
            'name': 'YuNet (Face Detector)',
            'url': 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx',
            'path': YUNET_MODEL_PATH,
            'size': '~2.8 MB'
        })
    
    if not os.path.exists(SFACE_MODEL_PATH):
        models_to_download.append({
            'name': 'SFace (Face Recognizer)',
            'url': 'https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx',
            'path': SFACE_MODEL_PATH,
            'size': '~10 MB'
        })
    
    if models_to_download:
        print("\n" + "═" * 60)
        print("  DEEP LEARNING MODELS REQUIRED")
        print("═" * 60)
        print("  Downloading models (one-time setup)...\n")
        
        for model in models_to_download:
            print(f"  → {model['name']} ({model['size']})... ", end='', flush=True)
            try:
                urllib.request.urlretrieve(model['url'], model['path'])
                print("✓ Downloaded")
            except Exception as e:
                print(f"✗ FAILED\n  ERROR: {e}")
                return False
        
        print("\n  ✓ All models downloaded successfully!")
        print("═" * 60 + "\n")
    return True

# ══════════════════════════════════════════════════════════════════════════════
# Custom Settings Override
# ══════════════════════════════════════════════════════════════════════════════
import json
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")
try:
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            _overrides = json.load(f)
            for _k, _v in _overrides.items():
                if _k in globals():
                    globals()[_k] = _v
except Exception as e:
    print(f"Error loading settings.json: {e}")

# Ensure all directories exist
for _dir in [DATA_DIR, CRIMINAL_DB_DIR, CAPTURED_DIR, TRAINING_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# Auto-download models
_models_ready = download_models_if_needed()

print("\n" + "=" * 60)
print("  AI FACE RECOGNITION SYSTEM — Configuration Loaded")
print("=" * 60)
print(f"  Engine Mode        : SFACE")
print(f"  Camera Resolution  : {FRAME_WIDTH}×{FRAME_HEIGHT}")
print(f"  Enrollment Images  : {ENROLL_FRAME_COUNT} per person")
print("=" * 60 + "\n")