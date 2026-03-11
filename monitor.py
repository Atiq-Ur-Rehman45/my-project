"""
==============================================================
  AI Face Recognition System — Live Recognition Monitor
  Real-time webcam face recognition with HUD overlay
==============================================================
"""

import cv2
import numpy as np
import os
import time
import logging
from datetime import datetime
from config import (
    FRAME_WIDTH, FRAME_HEIGHT, CAMERA_INDEX,
    COLOR_GREEN, COLOR_RED, COLOR_YELLOW, COLOR_WHITE,
    COLOR_BLACK, COLOR_ORANGE, COLOR_BLUE,
    ALERT_COOLDOWN_SECONDS, SNAPSHOT_ON_DETECTION,
    CAPTURED_DIR, RECOGNITION_CONFIDENCE_THRESHOLD, UNKNOWN_LABEL
)

logger = logging.getLogger(__name__)


class LiveMonitor:
    """
    Real-time face recognition monitor.
    
    Draws:
    - Green boxes for unknown faces
    - Red boxes for recognized criminals
    - HUD with stats, FPS, detection log
    - Confidence meter under each face
    """

    def __init__(self, face_engine, db_manager):
        self.engine      = face_engine
        self.db          = db_manager
        self.alert_log   = {}   # {criminal_id: last_alert_timestamp}
        self.fps_history = []
        self.detection_history = []   # Recent detections for HUD display
        self.max_history = 5

    # ── Main Loop ─────────────────────────────────────────────────────────────

    def run(self, camera_index=CAMERA_INDEX):
        """Start the live recognition feed."""
        if not self.engine.model_loaded:
            print("\n[MONITOR] ⚠ No trained model found.")
            print("[MONITOR]   → Enroll faces first (Option 1 in menu), then train (Option 2).\n")
            print("[MONITOR]   Running in DETECTION-ONLY mode (faces detected but not recognized).\n")

        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            raise RuntimeError(f"[MONITOR] Cannot open camera index {camera_index}")

        print("\n[MONITOR] Live feed started.")
        print("[MONITOR] Controls:")
        print("  Q / ESC → Quit")
        print("  S       → Save snapshot manually")
        print("  P       → Pause / Resume\n")

        paused    = False
        frame_num = 0
        prev_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[MONITOR] Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)

            if not paused:
                frame_num += 1
                current_time = time.time()
                fps = 1.0 / max(current_time - prev_time, 0.001)
                prev_time = current_time
                self.fps_history.append(fps)
                if len(self.fps_history) > 30:
                    self.fps_history.pop(0)
                avg_fps = sum(self.fps_history) / len(self.fps_history)

                # ── Recognition ───────────────────────────────────────────────
                results, gray = self.engine.recognize_all_faces(frame)

                for r in results:
                    self._draw_face_box(frame, r)
                    if r["is_known"]:
                        self._handle_alert(r, frame)

                # ── HUD ───────────────────────────────────────────────────────
                self._draw_hud(frame, avg_fps, len(results), frame_num)

            else:
                self._draw_paused_overlay(frame)

            cv2.imshow("AI Face Recognition System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):   # Q or ESC
                break
            elif key == ord('s') or key == ord('S'):
                self._save_snapshot(frame, tag="manual")
            elif key == ord('p') or key == ord('P'):
                paused = not paused
                state = "PAUSED" if paused else "RESUMED"
                print(f"[MONITOR] {state}")

        cap.release()
        cv2.destroyAllWindows()
        print("[MONITOR] Feed closed.")

    # ── Drawing Helpers ───────────────────────────────────────────────────────

    def _draw_face_box(self, frame, r):
        """Draw bounding box, name label, and confidence bar for a face."""
        x, y, w, h    = r["x"], r["y"], r["w"], r["h"]
        name          = r["name"]
        confidence    = r["confidence"]
        is_known      = r["is_known"]

        # Box color: RED for known criminal, GREEN for unknown
        box_color = COLOR_RED if is_known else COLOR_GREEN
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

        # Corner accents (stylized box corners)
        corner_len = 15
        corner_t   = 3
        for (cx, cy) in [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]:
            dx = 1 if cx == x else -1
            dy = 1 if cy == y else -1
            cv2.line(frame, (cx, cy), (cx + dx * corner_len, cy), box_color, corner_t)
            cv2.line(frame, (cx, cy), (cx, cy + dy * corner_len), box_color, corner_t)

        # Name label background
        label_text = f"{name}"
        if is_known:
            label_text += f"  [{confidence:.1f}%match]"
            criminal = r.get("criminal", {})
            if criminal:
                crime = criminal.get("crime_type", "")
                if crime:
                    label_text += f"  Crime: {crime}"

        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_y = y - 10 if y > 40 else y + h + 20

        cv2.rectangle(frame, (x, label_y - th - 6), (x + tw + 8, label_y + 4), box_color, -1)
        cv2.putText(frame, label_text, (x + 4, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_BLACK, 1)

        # Confidence meter bar (only for recognized criminals)
        if is_known:
            meter_y    = y + h + 8
            max_dist   = RECOGNITION_CONFIDENCE_THRESHOLD
            match_pct  = max(0, 1.0 - confidence / max_dist)
            bar_width  = int(w * match_pct)
            cv2.rectangle(frame, (x, meter_y), (x + w, meter_y + 5), (60, 60, 60), -1)
            meter_color = COLOR_GREEN if match_pct > 0.6 else COLOR_YELLOW if match_pct > 0.3 else COLOR_RED
            cv2.rectangle(frame, (x, meter_y), (x + bar_width, meter_y + 5), meter_color, -1)

        # ALERT flash for known criminals
        if is_known:
            flash_alpha = 0.15
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), COLOR_RED, -1)
            cv2.addWeighted(overlay, flash_alpha, frame, 1 - flash_alpha, 0, frame)

    def _draw_hud(self, frame, fps, face_count, frame_num):
        """Draw the top HUD bar with system info."""
        h, w = frame.shape[:2]

        # Top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # System title
        cv2.putText(frame, "AI FACE RECOGNITION SYSTEM", (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_ORANGE, 1)

        # FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GREEN, 1)

        # Face count
        faces_text = f"Faces: {face_count}"
        cv2.putText(frame, faces_text, (120, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)

        # Timestamp
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        (tw, _), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(frame, ts, (w - tw - 10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)

        # Model status
        status  = "MODEL: ACTIVE" if self.engine.model_loaded else "MODEL: NOT TRAINED"
        s_color = COLOR_GREEN if self.engine.model_loaded else COLOR_RED
        (sw, _), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(frame, status, (w - sw - 10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, s_color, 1)

        # Bottom: recent detections log
        if self.detection_history:
            log_y = h - 10
            for log_entry in reversed(self.detection_history[-3:]):
                cv2.putText(frame, log_entry, (10, log_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_RED, 1)
                log_y -= 18

        # Controls reminder
        ctrl_text = "[Q/ESC] Quit  [S] Snapshot  [P] Pause"
        (cw, _), _ = cv2.getTextSize(ctrl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(frame, ctrl_text, (w // 2 - cw // 2, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

    def _draw_paused_overlay(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        cv2.putText(frame, "PAUSED — Press P to resume", (80, frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_YELLOW, 2)

    # ── Alert / Logging ───────────────────────────────────────────────────────

    def _handle_alert(self, result, frame):
        """Log detection to DB, save snapshot, and print terminal alert."""
        criminal  = result.get("criminal")
        if not criminal:
            return

        criminal_id = criminal["id"]
        now         = time.time()

        # Respect cooldown to avoid flooding logs
        last_alert = self.alert_log.get(criminal_id, 0)
        if now - last_alert < ALERT_COOLDOWN_SECONDS:
            return

        self.alert_log[criminal_id] = now

        # Save snapshot
        snapshot_path = None
        if SNAPSHOT_ON_DETECTION:
            snapshot_path = self._save_snapshot(frame, tag=f"criminal_{criminal_id}")

        # DB log
        self.db.log_detection(
            criminal_id   = criminal_id,
            detected_name = result["name"],
            confidence    = result["confidence"],
            snapshot_path = snapshot_path
        )

        # Terminal alert
        self._print_alert(criminal, result["confidence"])

        # HUD log entry
        ts  = datetime.now().strftime("%H:%M:%S")
        msg = f"⚠ ALERT [{ts}] {result['name']} detected (conf: {result['confidence']:.1f})"
        self.detection_history.append(msg)
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)

    def _print_alert(self, criminal, confidence):
        """Print formatted terminal alert."""
        print("\n" + "!" * 60)
        print("  ⚠  CRIMINAL DETECTED!")
        print(f"  Name      : {criminal['name']}")
        print(f"  Crime     : {criminal.get('crime_type', 'N/A')}")
        print(f"  Status    : {criminal.get('status', 'N/A')}")
        print(f"  Confidence: {confidence:.2f} (lower = closer match)")
        print(f"  Time      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("!" * 60 + "\n")

    def _save_snapshot(self, frame, tag="snapshot"):
        """Save a JPEG snapshot to the captured_faces directory."""
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{tag}_{ts}.jpg"
        path     = os.path.join(CAPTURED_DIR, filename)
        cv2.imwrite(path, frame)
        print(f"[MONITOR] Snapshot saved → {path}")
        return path
