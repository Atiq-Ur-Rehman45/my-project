"""
web/services/enrollment_service.py
Web-adapted enrollment flow.
Instead of cv2.imshow(), frames are encoded as JPEG and exposed via MJPEG.
Instead of cv2.waitKey(), progress events are sent via WebSocket.
"""

import threading
import time
import logging
import cv2
import numpy as np
from collections import defaultdict
from datetime import datetime

from config import (
    FRAME_WIDTH, FRAME_HEIGHT, CAMERA_INDEX,
    MJPEG_QUALITY, ENROLLMENT_STRATEGY,
    ENROLL_CAPTURE_DELAY, ENROLL_COUNTDOWN, ENROLL_STAGE_PAUSE_SECONDS,
    ENROLL_FACE_MIN_AREA_RATIO, ENROLL_POSE_RELAX_AFTER_SECONDS,
    ENROLL_SAVE_FACE_CROPS, ENROLL_STAGE_TIMEOUT_SECONDS,
    ENROLL_LIVENESS_CHALLENGE_ENABLED, ENROLL_MIN_STABLE_FRAMES,
    ENABLE_BLUR_DETECTION, BLUR_THRESHOLD,
    COLOR_GREEN, COLOR_YELLOW, COLOR_WHITE, COLOR_ORANGE,
    COLOR_RED, COLOR_BLACK,
)

logger = logging.getLogger(__name__)


class EnrollmentService:
    """
    Runs the face enrollment pipeline in a background thread.
    Emits WebSocket progress events and streams MJPEG frames to the browser.
    """

    STATUS_IDLE       = "idle"
    STATUS_RUNNING    = "running"
    STATUS_COMPLETE   = "complete"
    STATUS_CANCELLED  = "cancelled"
    STATUS_ERROR      = "error"

    def __init__(self, face_engine, socketio):
        self.engine   = face_engine
        self.socketio = socketio

        self._lock      = threading.Lock()
        self._frame_jpeg = None
        self._running   = False
        self._thread    = None
        self.status     = self.STATUS_IDLE

        self.collected_frames = []   # BGR frames captured during enrollment
        self.save_dir  = None
        self.label     = None
        self._cancel_requested = False

        self._offline_jpeg = self._make_offline_jpeg()

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self, label: int, save_dir: str, camera_index: int = CAMERA_INDEX):
        """Begin enrollment capture in a background thread."""
        if self._running:
            return False
        self.label     = label
        self.save_dir  = save_dir
        self.collected_frames = []
        self._cancel_requested = False
        self.status    = self.STATUS_RUNNING

        self._thread = threading.Thread(
            target=self._enrollment_loop,
            args=(camera_index,),
            daemon=True
        )
        self._thread.start()
        return True

    def cancel(self):
        """Request enrollment cancellation."""
        self._cancel_requested = True

    def get_frame(self) -> bytes:
        """Return latest enrollment camera JPEG for MJPEG streaming."""
        with self._lock:
            return self._frame_jpeg or self._offline_jpeg

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Internal ─────────────────────────────────────────────────────────────

    def _enrollment_loop(self, camera_index: int):
        self._running = True
        strategy      = ENROLLMENT_STRATEGY
        total_target  = sum(s["count"] for s in strategy)

        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Warmup
        warmup_start = time.time()
        while time.time() - warmup_start < 2.0:
            cap.read()

        if not cap.isOpened():
            self.status = self.STATUS_ERROR
            self._running = False
            with self._lock:
                self._frame_jpeg = None
            self.socketio.emit("enrollment:error", {"message": "Cannot open camera"})
            return

        # Countdown
        countdown_start = time.time()
        while time.time() - countdown_start < ENROLL_COUNTDOWN:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            remaining = ENROLL_COUNTDOWN - int(time.time() - countdown_start)
            disp = frame.copy()
            cv2.putText(disp, str(remaining),
                        (FRAME_WIDTH//2 - 50, FRAME_HEIGHT//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 5.0, COLOR_ORANGE, 12)
            cv2.putText(disp, "Get Ready!",
                        (FRAME_WIDTH//2 - 100, FRAME_HEIGHT//2 + 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_WHITE, 2)
            self._store_frame(disp)
            if self._cancel_requested:
                break

        # Main collection loop
        current_angle_idx  = 0
        angle_count        = 0
        total_collected    = 0
        last_capture       = 0.0
        pause_until        = 0.0
        stage_start_time   = time.time()
        stage_stable_frames = 0
        front_pose_baseline = None
        reject_reasons      = defaultdict(int)

        try:
            while current_angle_idx < len(strategy) and not self._cancel_requested:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                disp  = frame.copy()
                now   = time.time()

                stage          = strategy[current_angle_idx]
                current_angle  = stage["angle"]
                target_count   = stage["count"]
                instruction    = stage["instruction"]
                stage_elapsed  = now - stage_start_time
                pose_relaxed   = stage_elapsed >= ENROLL_POSE_RELAX_AFTER_SECONDS
                liveness_ok    = (not ENROLL_LIVENESS_CHALLENGE_ENABLED)

                # Non-blocking inter-stage pause
                if now < pause_until:
                    self._draw_pause_overlay(disp, instruction, pause_until - now)
                    self._store_frame(disp)
                    continue

                # Detect & qualify face
                face_detected = quality_ok = pose_ok = False
                quality_message = ""
                x = y = w = h = 0
                yaw_val = pitch_val = None

                self.engine._set_sface_input_size((FRAME_WIDTH, FRAME_HEIGHT))
                _, faces = self.engine.detector.detect(frame)
                face_detected = faces is not None and len(faces) > 0
                if face_detected:
                    best = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                    x, y, w, h = best[0:4].astype(int)
                    face_area   = (w * h) / float(FRAME_WIDTH * FRAME_HEIGHT)
                    roi         = frame[max(0,y):y+h, max(0,x):x+w]
                    if len(faces) > 1:
                        quality_message = "ONE FACE ONLY"
                    elif face_area < ENROLL_FACE_MIN_AREA_RATIO:
                        quality_message = "MOVE CLOSER"
                    elif roi.size == 0:
                        quality_message = "FACE OUT OF FRAME"
                    else:
                        if ENABLE_BLUR_DETECTION:
                            gray_roi   = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                            blur_score = self.engine.estimate_blur(gray_roi)
                            if blur_score < BLUR_THRESHOLD:
                                quality_message = "HOLD STILL (BLURRY)"
                            else:
                                quality_ok = True
                        else:
                            quality_ok = True

                    if quality_ok:
                        yaw_val, pitch_val = self.engine._sface_pose_signature(best)
                        pose_ok = self.engine._pose_matches_stage(
                            current_angle, yaw_val, pitch_val, relaxed=pose_relaxed
                        )
                        if not pose_ok:
                            quality_message = f"ADJUST: {current_angle}"

                        if quality_ok and pose_ok and ENROLL_LIVENESS_CHALLENGE_ENABLED:
                            if current_angle.upper() == "FRONT":
                                liveness_ok = True
                            else:
                                liveness_ok = self.engine._liveness_pose_delta_ok(
                                    current_angle, yaw_val, pitch_val, front_pose_baseline
                                )
                                if not liveness_ok:
                                    quality_message = "LIVENESS: MOVE HEAD AS PROMPTED"
                        elif quality_ok and pose_ok:
                            liveness_ok = True
                else:
                    quality_message = "NO FACE DETECTED"

                if quality_ok and pose_ok and liveness_ok:
                    stage_stable_frames += 1
                else:
                    stage_stable_frames = 0

                can_capture = (
                    quality_ok and pose_ok and liveness_ok
                    and stage_stable_frames >= ENROLL_MIN_STABLE_FRAMES
                    and (now - last_capture) >= ENROLL_CAPTURE_DELAY
                )

                if can_capture:
                    self.collected_frames.append(frame.copy())
                    angle_count     += 1
                    total_collected += 1
                    last_capture     = now

                    if current_angle.upper() == "FRONT" and yaw_val:
                        if front_pose_baseline is None:
                            front_pose_baseline = {"yaw": yaw_val, "pitch": pitch_val}
                        else:
                            front_pose_baseline["yaw"]   = (front_pose_baseline["yaw"]   + yaw_val)   * 0.5
                            front_pose_baseline["pitch"] = (front_pose_baseline["pitch"] + pitch_val) * 0.5

                    # Save image to disk (Fix 5: Lossless PNG)
                    if self.save_dir:
                        img_path = f"{self.save_dir}/{current_angle}_{angle_count:02d}.png"
                        cv2.imwrite(img_path, frame)
                        if ENROLL_SAVE_FACE_CROPS and w > 0:
                            crop = frame[max(0,y):y+h, max(0,x):x+w]
                            if crop.size > 0:
                                cv2.imwrite(f"{self.save_dir}/{current_angle}_{angle_count:02d}_crop.png", crop)

                    # Emit progress via WebSocket
                    self.socketio.emit("enrollment:progress", {
                        "stage":          current_angle,
                        "captured":       angle_count,
                        "target":         target_count,
                        "total_captured": total_collected,
                        "total_target":   total_target,
                        "quality_message": "",
                        "all_stages": [
                            {
                                "angle":    s["angle"],
                                "target":   s["count"],
                                "captured": (angle_count if i == current_angle_idx
                                             else (s["count"] if i < current_angle_idx else 0)),
                            }
                            for i, s in enumerate(strategy)
                        ],
                    })

                    if angle_count >= target_count:
                        current_angle_idx += 1
                        angle_count        = 0
                        stage_stable_frames = 0
                        if current_angle_idx < len(strategy):
                            pause_until      = now + ENROLL_STAGE_PAUSE_SECONDS
                            stage_start_time = now
                            self.socketio.emit("enrollment:stage_change", {
                                "new_stage":   strategy[current_angle_idx]["angle"],
                                "instruction": strategy[current_angle_idx]["instruction"],
                            })
                else:
                    if now - last_capture >= ENROLL_CAPTURE_DELAY:
                        reject_reasons["no_capture"] += 1

                # Draw overlay on camera frame
                self._draw_enrollment_hud(
                    disp, current_angle, instruction,
                    angle_count, target_count, total_collected, total_target,
                    face_detected, quality_ok, pose_ok, x, y, w, h, quality_message,
                    pose_relaxed, yaw_val, pitch_val, stage_elapsed
                )
                self._store_frame(disp)
        finally:
            cap.release()

        with self._lock:
            self._frame_jpeg = None

        if self._cancel_requested:
            self.status = self.STATUS_CANCELLED
            self.socketio.emit("enrollment:cancelled", {})
        else:
            self.status = self.STATUS_COMPLETE
            self.socketio.emit("enrollment:complete", {
                "total_images": total_collected,
                "label":        self.label,
            })

        self._running = False


    def _store_frame(self, frame):
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, MJPEG_QUALITY])
        with self._lock:
            self._frame_jpeg = jpeg.tobytes()

    def _draw_enrollment_hud(self, disp, angle, instruction, angle_count, target_count,
                              total_collected, total_target, face_detected,
                              quality_ok, pose_ok, x, y, w, h, quality_message,
                              pose_relaxed, yaw_val, pitch_val, stage_elapsed):
        # Top bar
        cv2.rectangle(disp, (0, 0), (FRAME_WIDTH, 110), (20, 25, 35), -1)
        cv2.putText(disp, f"ANGLE: {angle}", (14, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_YELLOW, 2)
        cv2.putText(disp, instruction, (14, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, COLOR_WHITE, 1)
        cv2.putText(disp, f"{angle_count}/{target_count}  Total:{total_collected}/{total_target}",
                    (14, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_GREEN, 1)
        if pose_relaxed:
            cv2.putText(disp, "POSE ASSIST ON", (FRAME_WIDTH - 200, 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_YELLOW, 1)
        if stage_elapsed >= ENROLL_STAGE_TIMEOUT_SECONDS:
            cv2.putText(disp, "ADJUST LIGHT/ANGLE", (FRAME_WIDTH - 240, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_YELLOW, 1)

        # Face box
        if face_detected and w > 0:
            box_color = COLOR_GREEN if (quality_ok and pose_ok) else COLOR_YELLOW
            cv2.rectangle(disp, (x, y), (x+w, y+h), box_color, 2)
            if yaw_val is not None:
                cv2.putText(disp, f"Y:{yaw_val:.2f} P:{pitch_val:.2f}",
                            (x, max(18, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_WHITE, 1)

        # Quality message
        if quality_message:
            (tw, th), _ = cv2.getTextSize(quality_message, cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2)
            cx = (FRAME_WIDTH - tw) // 2
            cy = FRAME_HEIGHT - 60
            cv2.rectangle(disp, (cx-8, cy-th-4), (cx+tw+8, cy+6), (30, 30, 30), -1)
            cv2.putText(disp, quality_message, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, COLOR_YELLOW, 2)

        if not face_detected:
            cv2.putText(disp, "NO FACE DETECTED",
                        (FRAME_WIDTH//2 - 145, FRAME_HEIGHT - 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_RED, 2)

        # Bottom progress bar
        progress = total_collected / max(total_target, 1)
        bar_h    = FRAME_HEIGHT - 18
        bar_w    = int((FRAME_WIDTH - 40) * progress)
        cv2.rectangle(disp, (20, bar_h), (FRAME_WIDTH-20, bar_h+12), (50, 50, 50), -1)
        cv2.rectangle(disp, (20, bar_h), (20 + bar_w, bar_h+12), COLOR_GREEN, -1)

    def _draw_pause_overlay(self, disp, instruction, remaining):
        overlay = disp.copy()
        cv2.rectangle(overlay, (0, 0), (FRAME_WIDTH, FRAME_HEIGHT), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, disp, 0.55, 0, disp)
        cv2.putText(disp, "CHANGING POSE",
                    (FRAME_WIDTH//2 - 155, FRAME_HEIGHT//2 - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, COLOR_YELLOW, 3)
        cv2.putText(disp, instruction,
                    (FRAME_WIDTH//2 - 200, FRAME_HEIGHT//2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, COLOR_WHITE, 2)
        cv2.putText(disp, f"{int(remaining)+1}...",
                    (FRAME_WIDTH//2 - 25, FRAME_HEIGHT//2 + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, COLOR_ORANGE, 4)

    def _make_offline_jpeg(self) -> bytes:
        img = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        img[:] = (15, 18, 25)
        cv2.putText(img, "ENROLLMENT CAMERA", (FRAME_WIDTH//2 - 175, FRAME_HEIGHT//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (55, 70, 90), 2)
        cv2.putText(img, "Fill in the form and click Start Enrollment",
                    (FRAME_WIDTH//2 - 270, FRAME_HEIGHT//2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 55, 70), 1)
        _, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return jpeg.tobytes()
