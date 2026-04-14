
import os
import sys
import logging
from datetime import datetime

# ── Setup logging ─────────────────────────────────────────────────────────────
from config import LOG_FILE, LOGS_DIR
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
# Suppress verbose OpenCV logs
logging.getLogger("cv2").setLevel(logging.ERROR)
logger = logging.getLogger("main")


# ── Imports ───────────────────────────────────────────────────────────────────
from database import DatabaseManager
from face_engine import FaceEngine
from trainer import TrainingManager
from monitor import LiveMonitor
from config import ENROLL_FRAME_COUNT, CAMERA_INDEX


# ── Banner ────────────────────────────────────────────────────────────────────
BANNER = r"""
+--------------------------------------------------------------+
|      AI-BASED FACE & WEAPON RECOGNITION SYSTEM               |
|      Module: Face Recognition                                |
+--------------------------------------------------------------+
"""

MENU = """
+------------------------------+
|         MAIN MENU            |
+------------------------------+
|  1. Enroll New Criminal      |
|  2. Train / Retrain Model    |
|  3. Start Live Recognition   |
|  3.5 Test Video File         |
|  4. View All Records         |
|  5. View Detection Logs      |
|  6. Delete Criminal Record   |
|  7. System Status            |
|  0. Exit                     |
+------------------------------+
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def print_separator(char="-", width=60):
    print(char * width)

def input_required(prompt):
    while True:
        val = input(prompt).strip()
        if val:
            return val
        print("  ✗ This field is required.")

def confirm(prompt):
    return input(f"{prompt} [y/N]: ").strip().lower() == "y"


# ── Menu Actions ──────────────────────────────────────────────────────────────

def enroll_criminal(db, engine, trainer):
    """Enroll a new person into the criminal database."""
    print("\n" + "=" * 55)
    print("  ENROLL NEW CRIMINAL")
    print("=" * 55)

    name       = input_required("  Full Name        : ")
    cnic       = input("  CNIC (optional)  : ").strip() or None
    crime_type = input("  Crime Type       : ").strip() or "Unknown"
    status     = input("  Status [Wanted]  : ").strip() or "Wanted"
    notes      = input("  Notes (optional) : ").strip() or None

    print(f"\n  Preview:")
    print(f"  Name      → {name}")
    print(f"  CNIC      → {cnic or 'N/A'}")
    print(f"  Crime     → {crime_type}")
    print(f"  Status    → {status}")

    if not confirm("\n  Confirm and capture face?"):
        print("  ✗ Enrollment cancelled.")
        return

    # Add DB record first
    criminal_id, face_label, image_dir = db.add_criminal(
        name=name, cnic=cnic, crime_type=crime_type,
        status=status, notes=notes
    )

    print(f"\n  [DB] Record created — ID: {criminal_id}, Face Label: {face_label}")
    print(f"  [CAM] Opening webcam to collect face samples...")
    print(f"  [CAM] Will capture {ENROLL_FRAME_COUNT} frames across multiple angles.")
    print(f"  [CAM] Follow on-screen instructions for best results.\n")

    input("  Press ENTER to open camera...")

    try:
        samples = engine.collect_face_samples(
            camera_index = CAMERA_INDEX,
            target_count = ENROLL_FRAME_COUNT,
            label        = face_label,
            save_dir     = image_dir
        )
    except RuntimeError as e:
        print(f"\n  ✗ Camera error: {e}")
        print("  → Check your webcam connection.")
        return
    except Exception as e:
        print(f"\n  ✗ Unexpected error: {e}")
        return

    if len(samples) < 10:
        print(f"\n  ⚠ Only {len(samples)} samples collected. Consider re-enrolling for better accuracy.")
    else:
        print(f"\n  ✓ {len(samples)} face samples collected and saved to:")
        print(f"    {image_dir}")

    print("\n  ✓ Enrollment complete!")
    print(f"  ⚡ Run Option 2 (Train Model) to activate recognition for {name}.\n")


def train_model(db, engine, trainer):
    """Generate face embeddings for all enrolled criminals using SFace."""
    print("\n" + "=" * 55)
    print("  GENERATE FACE EMBEDDINGS (TRAINING)")
    print("=" * 55)

    criminals = db.list_all_criminals()
    if not criminals:
        print("  ✗ No criminals enrolled yet. Use Option 1 first.")
        return

    trainer.get_training_summary()

    if not confirm("  Start training now?"):
        print("  ✗ Training cancelled.")
        return

    print()
    start = datetime.now()
    success = trainer.full_retrain()
    elapsed = (datetime.now() - start).total_seconds()

    if success:
        print(f"\n  ✓ Training completed in {elapsed:.2f}s")
        print(f"  ✓ SFace database updated — ready for live recognition.\n")
    else:
        print(f"\n  ✗ Training failed. Check that enrolled persons have face images.\n")


def start_live_recognition(engine, db, monitor):
    """Launch the live webcam recognition feed."""
    print("\n" + "=" * 55)
    print("  LIVE RECOGNITION MONITOR")
    print("=" * 55)

    if not engine.model_loaded:
        print("  ⚠ No trained model detected.")
        print("  → Faces will be detected but NOT identified.")
        if not confirm("  Continue in detection-only mode?"):
            return

    # Refresh label map before starting
    label_map = db.get_label_criminal_map()
    engine.update_label_map(label_map)
    total_enrolled = len(label_map)
    print(f"\n  Monitoring for {total_enrolled} enrolled criminal(s).")
    print(f"  Controls: Q/ESC=Quit  S=Snapshot  P=Pause\n")
    input("  Press ENTER to start feed...")

    try:
        monitor.run(camera_index=CAMERA_INDEX)
    except RuntimeError as e:
        print(f"\n  ✗ Camera error: {e}")
        print("  → Check webcam connection and index in config.py (CAMERA_INDEX).\n")
    except Exception as e:
        print(f"\n  ✗ Unexpected error: {e}\n")


def view_all_records(db):
    """Display all criminal records in a formatted table."""
    print("\n" + "=" * 70)
    print("  CRIMINAL DATABASE RECORDS")
    print("=" * 70)
    criminals = db.list_all_criminals()

    if not criminals:
        print("  No records found.\n")
        return

    header = f"  {'ID':<5} {'Name':<22} {'Crime Type':<18} {'Status':<12} {'CNIC':<16}"
    print(header)
    print("  " + "-" * 68)

    for c in criminals:
        # Handle None values properly
        cid = c.get('id', 'N/A')
        name = c.get('name') or 'N/A'
        crime = c.get('crime_type') or 'N/A'
        status = c.get('status') or 'N/A'
        cnic = c.get('cnic') or 'N/A'
        
        print(f"  {cid:<5} {name:<22} {crime:<18} {status:<12} {cnic:<16}")

    print("=" * 70)
    print(f"  Total: {len(criminals)} record(s)\n")


def view_detection_logs(db):
    """Show recent detection events."""
    print("\n" + "=" * 70)
    print("  RECENT DETECTION LOGS")
    print("=" * 70)
    logs = db.get_recent_detections(limit=20)

    if not logs:
        print("  No detection events recorded yet.\n")
        return

    for log in logs:
        ts = log.get("timestamp") or "N/A"
        name = log.get("detected_name") or "Unknown"
        conf = log.get("confidence", 0.0)
        cam = log.get("camera_id") or "N/A"
        print(f"  [{ts}]  {name:<20}  Conf: {conf:<7.2f}  Camera: {cam}")

    print("=" * 70)
    print(f"  Face events  : {db.get_detection_count()}")
    print(f"  Weapon events: {db.get_weapon_detection_count()}\n")


def delete_criminal(db, trainer, engine):
    """Delete a criminal record and always re-train to prevent stale identities."""
    print("\n" + "=" * 55)
    print("  DELETE CRIMINAL RECORD")
    print("=" * 55)

    view_all_records(db)

    try:
        cid_input = input("  Enter Criminal ID to delete (0 to cancel): ").strip()
        if not cid_input:
            print("  ✗ Invalid input.")
            return
        cid = int(cid_input)
    except ValueError:
        print("  ✗ Invalid input. Please enter a number.")
        return

    if cid == 0:
        print("  Cancelled.")
        return

    record = db.get_criminal_by_id(cid)
    if not record:
        print(f"  ✗ No criminal with ID {cid} found.")
        return

    name = record.get('name') or 'Unknown'
    print(f"\n  Will delete: {name} (ID: {cid})")
    if not confirm("  Are you sure?"):
        print("  Cancelled.")
        return

    purge_snaps = confirm("  Also delete this person's detection snapshots from captured_faces?")
    result = db.delete_criminal(cid, purge_snapshots=purge_snaps)

    if not result.get("deleted"):
        print("  ✗ Delete failed.")
        return

    print("  ✓ Record, related logs, and training images removed.")

    print("\n  Updating face embeddings to remove deleted identity...")
    success = trainer.full_retrain()

    # Always refresh runtime label map after delete.
    label_map = db.get_label_criminal_map()
    engine.update_label_map(label_map)

    if success:
        print("  ✓ Model updated and label map refreshed.\n")
    else:
        print("  ⚠ Re-train had no data or failed; label map refreshed, recognition model may be empty.\n")


def system_status(db, engine):
    """Print system health and statistics."""
    print("\n" + "=" * 55)
    print("  SYSTEM STATUS")
    print("=" * 55)

    criminals = db.list_all_criminals()
    total_images = 0
    for c in criminals:
        idir = c.get("image_dir") or ""
        if idir and os.path.isdir(idir):
            total_images += len([f for f in os.listdir(idir) if f.endswith((".jpg", ".png"))])

    print(f"  Database       : ✓ Connected")
    print(f"  Enrolled People: {len(criminals)}")
    print(f"  Total Images   : {total_images}")
    print(f"  Model Trained  : {'✓ YES' if engine.model_loaded else '✗ NO — train first'}")
    print(f"  Face Logs      : {db.get_detection_count()} events")
    print(f"  Weapon Logs    : {db.get_weapon_detection_count()} events")
    print(f"  Log File       : {LOG_FILE}")

    from config import CRIMINAL_DB_DIR, CAPTURED_DIR, SFACE_DB_PATH
    print(f"\n  Paths:")
    print(f"    Criminal DB   → {CRIMINAL_DB_DIR}")
    print(f"    Captures      → {CAPTURED_DIR}")
    print(f"    Embeddings    → {SFACE_DB_PATH}")
    print("=" * 55 + "\n")


def test_video_recognition(engine, db, monitor):
    """Test face and weapon recognition on a video file (MP4, AVI, etc)."""
    print("\n" + "=" * 55)
    print("  VIDEO FILE RECOGNITION TEST")
    print("=" * 55)
    print("  Test weapon detection without real weapons!")
    print("  Use downloaded YouTube MP4s or any local video file.\n")

    if not engine.model_loaded:
        print("  ⚠ No trained face model detected.")
        print("  → Face recognition disabled; weapon detection will run.\n")
        if not confirm("  Continue in weapon-detection-only mode?"):
            return

    # Refresh label map before starting
    label_map = db.get_label_criminal_map()
    engine.update_label_map(label_map)
    total_enrolled = len(label_map)
    print(f"\n  Monitoring for {total_enrolled} enrolled criminal(s).")

    # Get video file path
    video_path = input("  Enter video file path (or filename if in current directory): ").strip()
    if not video_path:
        print("  ✗ No path provided. Cancelled.\n")
        return

    if not os.path.isfile(video_path):
        print(f"  ✗ File not found: {video_path}\n")
        return

    print(f"\n  Loading video: {video_path}")
    print(f"  Controls: Q/ESC=Quit  S=Snapshot  P=Pause\n")
    input("  Press ENTER to start playback...")

    try:
        monitor.run(video_path=video_path)
    except RuntimeError as e:
        print(f"\n  ✗ Video error: {e}")
        print("  → Check file format and path.\n")
    except Exception as e:
        print(f"\n  ✗ Unexpected error: {e}\n")


# ── Main Entry Point ──────────────────────────────────────────────────────────

def main():
    print(BANNER)

    # Initialize components
    print("[BOOT] Initializing system components...")
    try:
        db      = DatabaseManager()
        engine  = FaceEngine()
        trainer = TrainingManager(engine, db)
        monitor = LiveMonitor(engine, db)
    except Exception as e:
        print(f"\n[BOOT] ✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Load label map into engine (needed for recognition)
    label_map = db.get_label_criminal_map()
    engine.update_label_map(label_map)

    print(f"[BOOT] ✓ System ready. Enrolled: {len(label_map)} person(s).")

    # ── Main Loop ─────────────────────────────────────────────────────────────
    while True:
        print(MENU)
        choice = input("  Select option: ").strip()

        if choice == "1":
            enroll_criminal(db, engine, trainer)

        elif choice == "2":
            train_model(db, engine, trainer)

        elif choice == "3":
            start_live_recognition(engine, db, monitor)

        elif choice == "3.5":
            test_video_recognition(engine, db, monitor)

        elif choice == "4":
            view_all_records(db)

        elif choice == "5":
            view_detection_logs(db)

        elif choice == "6":
            delete_criminal(db, trainer, engine)

        elif choice == "7":
            system_status(db, engine)

        elif choice == "0":
            print("\n  Goodbye! Stay safe.\n")
            break

        else:
            print(f"\n  ✗ Invalid option: '{choice}'. Choose 0–7 or 3.5.\n")


if __name__ == "__main__":
    main()