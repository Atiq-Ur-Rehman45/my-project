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
SFACE_MATCH_THRESHOLD = 0.363       # Cosine similarity threshold
                                     # 0.30-0.35 = Lenient (more matches, some false positives)
                                     # 0.363     = Recommended (balanced)
                                     # 0.40-0.45 = Strict (fewer false positives, may miss some)
                                     # 0.50+     = Very strict (high security mode)

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
FRAME_WIDTH           = 1280        # Camera resolution width
FRAME_HEIGHT          = 720         # Camera resolution height
                                     # Recommended: 1280×720 (good balance)
                                     # J5Create JVCU100 supports up to 1920×1080
CAMERA_INDEX          = 0           # 0 = default webcam, 1 = external

# ── Enrollment Settings ───────────────────────────────────────────────────────
# CRITICAL: SFace needs fewer images than LBPH!
if RECOGNITION_ENGINE == "SFACE":
    ENROLL_FRAME_COUNT = 10         # SFace: 10 images is enough
else:
    ENROLL_FRAME_COUNT = 30         # LBPH: needs 30 for good accuracy

ENROLL_CAPTURE_DELAY  = 0.4         # Seconds between auto-captures
ENROLL_COUNTDOWN      = 3           # Countdown before enrollment starts

# ── Multi-Angle Enrollment Strategy ───────────────────────────────────────────
if RECOGNITION_ENGINE == "SFACE":
    # SFace: Fewer images, emphasis on quality and variety
    ENROLLMENT_STRATEGY = [
        {"angle": "FRONT", "count": 4, "instruction": "Look STRAIGHT at camera"},
        {"angle": "LEFT",  "count": 2, "instruction": "Turn head to the LEFT"},
        {"angle": "RIGHT", "count": 2, "instruction": "Turn head to the RIGHT"},
        {"angle": "UP",    "count": 1, "instruction": "Tilt head UP slightly"},
        {"angle": "DOWN",  "count": 1, "instruction": "Tilt head DOWN slightly"},
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
                                     # Higher = stricter (100+)
                                     # Lower = more lenient (50-80)

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

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = os.path.join(LOGS_DIR, "system.log")

# ══════════════════════════════════════════════════════════════════════════════
# Auto-Download Models (SFace Only)
# ══════════════════════════════════════════════════════════════════════════════

def download_models_if_needed():
    """
    Auto-download YuNet and SFace models if not present.
    Called automatically when RECOGNITION_ENGINE = "SFACE"
    """
    if RECOGNITION_ENGINE != "SFACE":
        return True
    
    import urllib.request
    
    models_to_download = []
    
    # Check YuNet model
    if not os.path.exists(YUNET_MODEL_PATH):
        models_to_download.append({
            'name': 'YuNet (Face Detector)',
            'url': 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx',
            'path': YUNET_MODEL_PATH,
            'size': '~2.8 MB'
        })
    
    # Check SFace model
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
        print("  SFace mode requires YuNet and SFace ONNX models.")
        print("  Downloading models (one-time setup)...\n")
        
        for model in models_to_download:
            print(f"  → {model['name']} ({model['size']})... ", end='', flush=True)
            try:
                urllib.request.urlretrieve(model['url'], model['path'])
                print("✓ Downloaded")
            except Exception as e:
                print(f"✗ FAILED")
                print(f"\n  ERROR: {e}")
                print(f"  Please download manually from:")
                print(f"  {model['url']}")
                print(f"  Save to: {model['path']}\n")
                return False
        
        print("\n  ✓ All models downloaded successfully!")
        print("═" * 60 + "\n")
    
    return True

# ══════════════════════════════════════════════════════════════════════════════
# Initialization
# ══════════════════════════════════════════════════════════════════════════════

# Ensure all directories exist
for _dir in [DATA_DIR, CRIMINAL_DB_DIR, CAPTURED_DIR, TRAINING_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# Auto-download models if using SFace
if RECOGNITION_ENGINE == "SFACE":
    _models_ready = download_models_if_needed()
    if not _models_ready:
        print("\n⚠ WARNING: Model download failed!")
        print("   System will attempt to use models, but may fail.")
        print("   Consider switching to LBPH mode or downloading models manually.\n")

# ══════════════════════════════════════════════════════════════════════════════
# Configuration Summary (Printed on Import)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("  AI FACE RECOGNITION SYSTEM — Configuration Loaded")
print("═" * 60)
print(f"  Engine Mode        : {RECOGNITION_ENGINE}")
print(f"  Camera Resolution  : {FRAME_WIDTH}×{FRAME_HEIGHT}")
print(f"  Enrollment Images  : {ENROLL_FRAME_COUNT} per person")

if RECOGNITION_ENGINE == "SFACE":
    print(f"  Detection Method   : YuNet (Deep Neural Network)")
    print(f"  Recognition Method : SFace (Cosine Similarity)")
    print(f"  Match Threshold    : {SFACE_MATCH_THRESHOLD}")
    print(f"  Expected Accuracy  : 95-98%")
else:
    print(f"  Detection Method   : Haar Cascade")
    print(f"  Recognition Method : LBPH (Local Binary Patterns)")
    print(f"  Match Threshold    : {RECOGNITION_CONFIDENCE_THRESHOLD}")
    print(f"  Expected Accuracy  : 75-85%")

print("═" * 60 + "\n")