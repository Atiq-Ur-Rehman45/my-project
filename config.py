import os

# ── Base Paths ────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DATA_DIR         = os.path.join(BASE_DIR, "data")
CRIMINAL_DB_DIR  = os.path.join(DATA_DIR, "criminal_db")
CAPTURED_DIR     = os.path.join(DATA_DIR, "captured_faces")
TRAINING_DIR     = os.path.join(DATA_DIR, "training_data")
UPLOAD_DIR       = os.path.join(DATA_DIR, "uploads")
MODELS_DIR       = os.path.join(BASE_DIR, "models")
LOGS_DIR         = os.path.join(BASE_DIR, "logs")

# ── Database ──────────────────────────────────────────────────────────────────
DB_PATH          = os.path.join(DATA_DIR, "criminal_records.db")

# ══════════════════════════════════════════════════════════════════════════════
# 🔥 ENGINE SELECTION — CHANGE THIS TO SWITCH MODES
# ══════════════════════════════════════════════════════════════════════════════
RECOGNITION_ENGINE = "SFACE"     # Options: "SFACE" or "LBPH"
                                 # SFACE = Modern (95%+ accuracy)
                                 # LBPH  = Classic (75-85% accuracy)

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
    SFACE_MATCH_THRESHOLD_ACQUIRE = 0.53 # Balanced default after field tuning
    SFACE_MATCH_THRESHOLD_MAINTAIN = 0.50
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

# ── Elite-Level Accuracy Refinements ─────────────────────────────────────────
SFACE_TOP_K_MATCH          = 3     # Average the top-N matches per person for robustness
SFACE_MATCH_FLOOR          = 0.35  # Hard floor for identification (ignore lower scores)
SFACE_SHARPEN_MIN_SIZE     = 80    # Apply USM if width/height is below this (px)
ENABLE_USM_DEBUG           = True  # Draw indicator when sharpening is active

# Multi-face stabilization cache
SFACE_CACHE_REUSE_FRAMES = 2        # Reuse last recognition for up to N frames if same face box persists
SFACE_CACHE_TTL_SECONDS = 0.7       # Drop stale cached tracks quickly to avoid ghost identities
SFACE_CACHE_IOU_THRESHOLD = 0.65    # Strict overlap needed to reuse a cached identity
SFACE_CACHE_MAX_TRACKS = 64         # Hard cap to keep cache operations bounded
SFACE_SKIP_RECOGNITION_FRAMES = 2   # Skip expensive DNN inference for cached tracks within N frames (FPS boost)

# ── Tier 2 Accuracy Bundle (Advanced Software Optimization) ──────────────────
SFACE_OUTLIER_THRESHOLD    = 0.40   # Max avg cosine distance between enrollment embeddings
SFACE_OUTLIER_MIN_SAMPLES  = 10     # Never prune a person below this many samples

# ══════════════════════════════════════════════════════════════════════════════
# LBPH (Classic) Configuration
# ══════════════════════════════════════════════════════════════════════════════

# ── Model Paths ───────────────────────────────────────────────────────────────
LBPH_MODEL_PATH  = os.path.join(MODELS_DIR, "lbph_face_model.xml")

# ── Cascade Classifier Paths ──────────────────────────────────────────────────
import cv2
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_PROFILE_CASCADE = cv2.data.haarcascades + "haarcascade_profileface.xml"
EYE_CASCADE_PATH  = cv2.data.haarcascades + "haarcascade_eye.xml"

# ── LBPH Detection Parameters (Live Monitoring) ───────────────────────────────
DETECTION_SCALE_FACTOR   = 1.05     # Lower = more sensitive (1.05-1.3)
DETECTION_MIN_NEIGHBORS  = 3        # Lower = more detections (3-7)
DETECTION_MIN_SIZE       = (50, 50) # Minimum face size in pixels

# Enrollment Detection (Stricter for quality)
ENROLL_SCALE_FACTOR   = 1.15
ENROLL_MIN_NEIGHBORS  = 7
ENROLL_MIN_SIZE       = (80, 80)

# ── LBPH Recognition Parameters ───────────────────────────────────────────────
RECOGNITION_CONFIDENCE_THRESHOLD = 100   # LBPH distance threshold
                                          # Lower = stricter (50-70)
                                          # Higher = more lenient (80-120)
                                          # Recommended: 100

# Recognition Stability
RECOGNITION_SMOOTHING_FRAMES = 5    # Average over last N frames
RECOGNITION_CONFIDENCE_ALPHA = 0.3  # Smoothing factor (0.0-1.0)

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
# CRITICAL: SFace needs fewer images than LBPH!
if RECOGNITION_ENGINE == "SFACE":
    ENROLL_FRAME_COUNT = 20         # SFace: 20 images improves identity separation
else:
    ENROLL_FRAME_COUNT = 30         # LBPH: needs 30 for good accuracy

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
if RECOGNITION_ENGINE == "SFACE":
    # SFace: Balanced quality and diversity for stronger embeddings
    ENROLLMENT_STRATEGY = [
        {"angle": "FRONT", "count": 8, "instruction": "Look STRAIGHT at camera"},
        {"angle": "LEFT",  "count": 4, "instruction": "Turn head to the LEFT"},
        {"angle": "RIGHT", "count": 4, "instruction": "Turn head to the RIGHT"},
        {"angle": "UP",    "count": 2, "instruction": "Tilt head UP slightly"},
        {"angle": "DOWN",  "count": 2, "instruction": "Tilt head DOWN slightly"},
    ]
else:
    # LBPH: More images for better training
    ENROLLMENT_STRATEGY = [
        {"angle": "FRONT", "count": 10, "instruction": "Look STRAIGHT at camera - HOLD STILL"},
        {"angle": "LEFT",  "count": 6,  "instruction": "Turn head to the LEFT - HOLD"},
        {"angle": "RIGHT", "count": 6,  "instruction": "Turn head to the RIGHT - HOLD"},
        {"angle": "UP",    "count": 4,  "instruction": "Tilt head UP slightly - HOLD"},
        {"angle": "DOWN",  "count": 4,  "instruction": "Tilt head DOWN slightly - HOLD"},
    ]

# ── Image Quality Settings ────────────────────────────────────────────────────
ENABLE_BLUR_DETECTION = True        # Reject blurry frames during enrollment
BLUR_THRESHOLD        = 80.0        # Laplacian variance threshold

# ══════════════════════════════════════════════════════════════════════════════
# Web Dashboard Settings
# ══════════════════════════════════════════════════════════════════════════════
WEB_HOST         = "0.0.0.0"   # Listen on all interfaces
WEB_PORT         = 5000        # Dashboard URL: http://localhost:5000
WEB_DEBUG        = True        # Set to False for production
WEB_SECRET_KEY   = "face_system_secret_key_1337"

# Streaming quality
MJPEG_QUALITY    = 70          # 1-100 (higher = better quality, lower = higher FPS)
MJPEG_MAX_FPS    = 30          # Limit server-side output rate

# Video Upload Settings
ALLOWED_VIDEO_EXT = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
UPLOAD_MAX_MB     = 500         # Allow larger video uploads for surveillance testing

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

# -- Weapon Detection (Async Worker) ------------------------------------------
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
WEAPON_WARMUP_FRAMES = 3              # Dummy inferences at startup
WEAPON_FRAME_SKIP = 1                 # Process every Nth frame (1 = all)

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
    "shotgun": "CRITICAL",
    "knife": "HIGH",
    "scissors": "MEDIUM",
}

# -- Detection Mode (Focus Mode) ----------------------------------------------
# "combined"    -> Face + Weapon run simultaneously (weapon on async thread)
# "face_only"   -> Weapon detection OFF, 100% CPU to face recognition
# "weapon_only" -> Face recognition OFF, 100% CPU to weapon detection
DETECTION_MODE_DEFAULT = "combined"

# -- Weapon Performance Profiles -----------------------------------------------
WEAPON_PERFORMANCE_PROFILE = "balanced"  # "speed" | "balanced" | "accuracy"

# -- Weapon Anti-Spoofing (100% independent from face anti-spoofing) -----------
WEAPON_ANTISPOOF_ENABLED = False         # Default OFF for testing
WEAPON_ANTISPOOF_THRESHOLD = 0.75        # "Real weapon" probability threshold
WEAPON_ANTISPOOF_DEPTH_WEIGHT = 0.50     # Fusion weight: MiDaS depth
WEAPON_ANTISPOOF_TEXTURE_WEIGHT = 0.30   # Fusion weight: LBP texture
WEAPON_ANTISPOOF_EDGE_WEIGHT = 0.20      # Fusion weight: Laplacian edge
MIDAS_MODEL_PATH = os.path.join(MODELS_DIR, "midas_v21_small.onnx")
MIDAS_INPUT_SIZE = (256, 256)
MIDAS_DOWNLOAD_URL = "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.onnx"

# -- Weapon Snapshots & Logging ------------------------------------------------
WEAPON_SNAPSHOT_DIR = os.path.join(DATA_DIR, "weapon_snapshots")
WEAPON_LOG_CSV = os.path.join(LOGS_DIR, "weapon_events.csv")
WEAPON_LOG_JSON = os.path.join(LOGS_DIR, "weapon_events.json")

# -- Load weapon_config.yaml overrides (if available) --------------------------
WEAPON_CONFIG_PATH = os.path.join(BASE_DIR, "weapon_config.yaml")
try:
    import yaml
    if os.path.exists(WEAPON_CONFIG_PATH):
        with open(WEAPON_CONFIG_PATH, "r", encoding="utf-8") as _wf:
            _wcfg = yaml.safe_load(_wf) or {}

        # Model overrides
        _model = _wcfg.get("model", {})
        if _model.get("path"):
            WEAPON_MODEL_PATH = os.path.join(BASE_DIR, _model["path"])
        if _model.get("input_size"):
            WEAPON_INPUT_SIZE = tuple(_model["input_size"])
        if _model.get("confidence_threshold") is not None:
            WEAPON_CONFIDENCE_THRESHOLD = float(_model["confidence_threshold"])
        if _model.get("nms_threshold") is not None:
            WEAPON_NMS_THRESHOLD = float(_model["nms_threshold"])
        if _model.get("warmup_frames") is not None:
            WEAPON_WARMUP_FRAMES = int(_model["warmup_frames"])

        # Class map override
        if _wcfg.get("classes"):
            WEAPON_CLASSES = {int(k): v for k, v in _wcfg["classes"].items()}
        if _wcfg.get("threat_levels"):
            WEAPON_THREAT_LEVELS = _wcfg["threat_levels"]

        # Performance profile
        if _wcfg.get("performance_profile"):
            WEAPON_PERFORMANCE_PROFILE = _wcfg["performance_profile"]
        _profiles = _wcfg.get("profiles", {})
        _active_profile = _profiles.get(WEAPON_PERFORMANCE_PROFILE, {})
        if _active_profile.get("input_size"):
            WEAPON_INPUT_SIZE = tuple(_active_profile["input_size"])
        if _active_profile.get("frame_skip") is not None:
            WEAPON_FRAME_SKIP = int(_active_profile["frame_skip"])
        if _active_profile.get("confidence_threshold") is not None:
            WEAPON_CONFIDENCE_THRESHOLD = float(_active_profile["confidence_threshold"])

        # Detection mode
        if _wcfg.get("detection_mode"):
            DETECTION_MODE_DEFAULT = _wcfg["detection_mode"]

        # Anti-spoofing
        _as = _wcfg.get("antispoof", {})
        if _as.get("enabled") is not None:
            WEAPON_ANTISPOOF_ENABLED = bool(_as["enabled"])
        if _as.get("threshold") is not None:
            WEAPON_ANTISPOOF_THRESHOLD = float(_as["threshold"])
        _depth = _as.get("depth", {})
        if _depth.get("weight") is not None:
            WEAPON_ANTISPOOF_DEPTH_WEIGHT = float(_depth["weight"])
        _texture = _as.get("texture", {})
        if _texture.get("weight") is not None:
            WEAPON_ANTISPOOF_TEXTURE_WEIGHT = float(_texture["weight"])
        _edge = _as.get("edge", {})
        if _edge.get("weight") is not None:
            WEAPON_ANTISPOOF_EDGE_WEIGHT = float(_edge["weight"])

        # Alerts
        _alerts = _wcfg.get("alerts", {})
        if _alerts.get("cooldown_seconds") is not None:
            WEAPON_ALERT_COOLDOWN_SECONDS = int(_alerts["cooldown_seconds"])
        if _alerts.get("snapshot_on_detection") is not None:
            WEAPON_SNAPSHOT_ON_DETECTION = bool(_alerts["snapshot_on_detection"])
        if _alerts.get("snapshot_dir"):
            WEAPON_SNAPSHOT_DIR = os.path.join(BASE_DIR, _alerts["snapshot_dir"])

except ImportError:
    pass  # PyYAML not installed — use Python defaults above
except Exception as _e:
    print(f"Warning: Could not load weapon_config.yaml: {_e}")

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = os.path.join(LOGS_DIR, "system.log")

# ══════════════════════════════════════════════════════════════════════════════
# Auto-Download Models (SFace Only)
# ══════════════════════════════════════════════════════════════════════════════

def download_models_if_needed():
    """Auto-download YuNet and SFace models if not present."""
    if RECOGNITION_ENGINE != "SFACE":
        return True
    
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
# Custom Settings Override (from Web Dashboard)
# ══════════════════════════════════════════════════════════════════════════════
import json
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")
_SETTINGS_BLOCKLIST = {"RECOGNITION_ENGINE"}   # Never allow UI to change the engine
try:
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            _overrides = json.load(f)
            for _k, _v in _overrides.items():
                if _k in _SETTINGS_BLOCKLIST:
                    continue
                if _k in globals():
                    globals()[_k] = _v
except Exception as e:
    print(f"Warning: Could not load settings.json: {e}")

# Ensure all directories exist
for _dir in [DATA_DIR, CRIMINAL_DB_DIR, CAPTURED_DIR, TRAINING_DIR, UPLOAD_DIR,
             MODELS_DIR, LOGS_DIR, WEAPON_SNAPSHOT_DIR]:
    os.makedirs(_dir, exist_ok=True)

# Auto-download models if using SFace
if RECOGNITION_ENGINE == "SFACE":
    _models_ready = download_models_if_needed()

print("\n" + "=" * 60)
print("  AI FACE RECOGNITION SYSTEM -- Configuration Loaded")
print("=" * 60)
print(f"  Engine Mode        : {RECOGNITION_ENGINE}")
print(f"  Detection Mode     : {DETECTION_MODE_DEFAULT}")
print(f"  Camera Resolution  : {FRAME_WIDTH}x{FRAME_HEIGHT}")
print(f"  Enrollment Images  : {ENROLL_FRAME_COUNT} per person")
print(f"  Weapon Anti-Spoof  : {'ON' if WEAPON_ANTISPOOF_ENABLED else 'OFF (Testing)'}")
print("=" * 60 + "\n")