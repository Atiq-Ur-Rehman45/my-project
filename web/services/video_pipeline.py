"""
web/services/video_pipeline.py
Server-side video processing bridge between OpenCV engines and the MJPEG stream.
Runs face recognition + weapon detection in a background thread.
Emits real-time WebSocket events for alerts and stats.
"""

import threading
import time
import logging
import cv2
import numpy as np
from datetime import datetime

from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    MJPEG_QUALITY, MJPEG_MAX_FPS,
    ASYNC_CAMERA_CAPTURE, WARMUP_FRAMES,
    ALERT_COOLDOWN_SECONDS, SNAPSHOT_ON_DETECTION, CAPTURED_DIR,
    ENABLE_WEAPON_DETECTION, WEAPON_ALERT_COOLDOWN_SECONDS,
    WEAPON_ALERT_ON_UNKNOWN, WEAPON_SNAPSHOT_ON_DETECTION,
    SFACE_MATCH_THRESHOLD,
    COLOR_GREEN, COLOR_RED, COLOR_YELLOW, COLOR_WHITE, COLOR_ORANGE,
    COLOR_BLACK, UNKNOWN_LABEL,
    CAMERA_FPS_TARGET, CAMERA_AUTOFOCUS, CAMERA_BUFFER_SIZE,
    UPLOAD_DIR, ENABLE_USM_DEBUG
)
from monitor import LatestFrameCamera, DirectCamera, VideoFileCamera

logger = logging.getLogger(__name__)

try:
    from weapon_engine import AsyncWeaponDetector, draw_weapon_detections
except Exception:
    AsyncWeaponDetector = None
    def draw_weapon_detections(frame, detections):
        return frame


class VideoPipeline:
    """
    Manages server-side video processing for web streaming.
    Thread-safe: the main Flask thread reads encoded frames;
    the background thread writes them.
    """

    STATUS_IDLE     = "idle"
    STATUS_STARTING = "starting"
    STATUS_RUNNING  = "running"
    STATUS_PAUSED   = "paused"
    STATUS_STOPPING = "stopping"

    def __init__(self, engine, db, socketio):
        self.engine   = engine
        self.db       = db
        self.socketio = socketio

        self._lock               = threading.Lock()
        self._latest_frame_jpeg  = None          # bytes for MJPEG stream
        self._status             = self.STATUS_IDLE
        self._paused             = False
        self._running            = False
        self._thread             = None
        self._native_window_active = False

        self._camera_stream  = None
        self._weapon_detector = None
        self._frame_num      = 0

        # Alert cooldown tracking
        self._alert_log        = {}   # criminal_id -> last_alert_ts
        self._weapon_alert_log = {}   # weapon_type -> last_alert_ts

        # Live stats (read by /api/status)
        self.stats = {
            "fps":       0.0,
            "faces":     0,
            "weapons":   0,
            "frame_num": 0,
            "source":    None,
        }

        # Placeholder JPEG shown when feed is offline
        self._offline_frame = self._make_offline_placeholder()

    # ── Public control API ────────────────────────────────────────────────────

    def start_live(self, camera_index: int = CAMERA_INDEX):
        """Start live camera feed."""
        if self._running:
            self.stop()
        self._status = self.STATUS_STARTING
        self.stats["source"] = f"camera:{camera_index}"
        camera_cls = LatestFrameCamera if ASYNC_CAMERA_CAPTURE else DirectCamera
        self._camera_stream = camera_cls(camera_index).start()
        self._start_thread()

    def start_video(self, video_path: str):
        """Start video-file feed."""
        if self._running:
            self.stop()
        self._status = self.STATUS_STARTING
        self.stats["source"] = f"file:{video_path}"
        self._camera_stream = VideoFileCamera(video_path).start()
        self._start_thread()

    def stop(self):
        """Stop the processing thread and release resources."""
        self._running = False
        self._status  = self.STATUS_STOPPING
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._cleanup()
        self._status = self.STATUS_IDLE
        self._latest_frame_jpeg = self._offline_frame
        self._native_window_active = False
        self.socketio.emit("status:feed_stopped", {})

    def pause(self):
        self._paused = True
        self._status = self.STATUS_PAUSED

    def resume(self):
        self._paused = False
        self._status = self.STATUS_RUNNING

    def toggle_native_window(self):
        with self._lock:
            self._native_window_active = getattr(self, "_native_window_active", False)
            self._native_window_active = not self._native_window_active
            return self._native_window_active

    def take_snapshot(self):
        """Save the current frame to disk and return the path."""
        raw = self._get_raw_frame()
        if raw is None:
            return None
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{CAPTURED_DIR}/manual_{ts}.jpg"
        cv2.imwrite(path, raw)
        return path

    def get_frame(self) -> bytes:
        """Return the latest JPEG bytes for MJPEG streaming."""
        with self._lock:
            return self._latest_frame_jpeg or self._offline_frame

    @property
    def status(self) -> str:
        return self._status

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Internal ─────────────────────────────────────────────────────────────

    def _start_thread(self):
        self._running = True
        self._paused  = False
        self._frame_num = 0
        self._native_window_active = True

        # Weapon detector
        if ENABLE_WEAPON_DETECTION and AsyncWeaponDetector is not None:
            self._weapon_detector = AsyncWeaponDetector().start()
            if not self._weapon_detector.model_loaded:
                logger.warning("[PIPELINE] Weapon model missing — weapon detection disabled.")
                self._weapon_detector = None

        # Refresh label map
        label_map = self.db.get_label_criminal_map()
        self.engine.update_label_map(label_map)

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        """Main processing loop running in the background thread."""
        is_video  = "file:" in (self.stats.get("source") or "")
        fps_times = []
        prev_time = time.time()
        last_stats_emit = 0.0

        # Camera warmup for live feeds
        if not is_video:
            logger.info("[PIPELINE] Warming up camera...")
            for _ in range(WARMUP_FRAMES):
                self._camera_stream.read()

        self._status = self.STATUS_RUNNING

        while self._running:
            ret, frame = self._camera_stream.read()
            if not ret:
                if is_video:
                    logger.info("[PIPELINE] Video file ended.")
                    self.socketio.emit("status:video_ended", {})
                    break
                time.sleep(0.005)
                continue

            # Mirror live camera frames
            if not is_video:
                frame = cv2.flip(frame, 1)

            if self._paused:
                self._encode_and_store(frame)
                time.sleep(0.03)
                continue

            self._frame_num += 1

            # ── Face recognition ──────────────────────────────────────────
            face_results, _ = self.engine.recognize_all_faces(frame)

            # ── Weapon detection (async) ──────────────────────────────────
            weapon_detections = []
            if self._weapon_detector:
                self._weapon_detector.submit_frame(frame, self._frame_num)
                result = self._weapon_detector.get_latest_result()
                weapon_detections = result.get("detections", [])

            # ── Draw overlays ─────────────────────────────────────────────
            self._draw_faces(frame, face_results)
            if weapon_detections:
                draw_weapon_detections(frame, weapon_detections)
            self._draw_hud(frame, face_results, weapon_detections)

            # ── Encode for MJPEG stream ───────────────────────────────────
            self._encode_and_store(frame)

            # ── Native Window Display ─────────────────────────────────────
            if getattr(self, "_native_window_active", False):
                cv2.imshow("Surveillance Feed (Native)", frame)
                cv2.waitKey(1)
                self._native_window_was_active = True
            else:
                if getattr(self, "_native_window_was_active", False):
                    try: cv2.destroyWindow("Surveillance Feed (Native)")
                    except: pass
                    self._native_window_was_active = False

            # ── Alerts ───────────────────────────────────────────────────
            for r in face_results:
                if r["is_known"]:
                    self._handle_face_alert(r, frame)
            if weapon_detections:
                self._handle_weapon_alert(weapon_detections, frame)

            # ── FPS stats ─────────────────────────────────────────────────
            now = time.time()
            fps = 1.0 / max(now - prev_time, 0.001)
            prev_time = now
            fps_times.append(fps)
            if len(fps_times) > 30:
                fps_times.pop(0)
            avg_fps = sum(fps_times) / len(fps_times)

            self.stats.update({
                "fps":       round(avg_fps, 1),
                "faces":     len(face_results),
                "weapons":   len(weapon_detections),
                "frame_num": self._frame_num,
            })

            # Emit stats every second (not every frame — saves bandwidth)
            if now - last_stats_emit >= 1.0:
                self.socketio.emit("status:fps_update", self.stats)
                last_stats_emit = now

            # Throttle to MJPEG_MAX_FPS
            target_interval = 1.0 / MJPEG_MAX_FPS
            elapsed = time.time() - now
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)

        self._cleanup()
        self._status = self.STATUS_IDLE

    def _cleanup(self):
        if self._camera_stream:
            self._camera_stream.stop()
            self._camera_stream = None
        if self._weapon_detector:
            self._weapon_detector.stop()
            self._weapon_detector = None
            
        try:
            cv2.destroyWindow("Surveillance Feed (Native)")
        except:
            pass
        self._native_window_was_active = False
        
        self.stats["source"] = None

    def _encode_and_store(self, frame):
        if getattr(self, "_native_window_active", False):
            if not hasattr(self, "_native_placeholder"):
                self._native_placeholder = self._make_offline_placeholder(True)
            with self._lock:
                self._latest_frame_jpeg = self._native_placeholder
            return
            
        _, jpeg = cv2.imencode(
            ".jpg", frame,
            [cv2.IMWRITE_JPEG_QUALITY, MJPEG_QUALITY]
        )
        with self._lock:
            self._latest_frame_jpeg = jpeg.tobytes()

    def _get_raw_frame(self):
        """Try to decode the latest JPEG back to a BGR frame for saving."""
        with self._lock:
            data = self._latest_frame_jpeg
        if data is None:
            return None
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _draw_faces(self, frame, results):
        for r in results:
            x, y, w, h = r["x"], r["y"], r["w"], r["h"]
            is_known   = r["is_known"]
            name       = r["name"]
            confidence = r["confidence"]

            box_color = COLOR_RED if is_known else COLOR_GREEN
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

            # Corner accents
            cl = 14
            for (cx, cy) in [(x, y), (x+w, y), (x, y+h), (x+w, y+h)]:
                dx = 1 if cx == x else -1
                dy = 1 if cy == y else -1
                cv2.line(frame, (cx, cy), (cx + dx*cl, cy), box_color, 3)
                cv2.line(frame, (cx, cy), (cx, cy + dy*cl), box_color, 3)

            # Label
            if is_known:
                pct = self.engine.remap_confidence(confidence) * 100
                pct = max(0.0, min(100.0, pct))
                label = f"{name}  [{pct:.0f}%]"
                criminal = r.get("criminal") or {}
                crime = criminal.get("crime_type", "")
                if crime:
                    label += f"  {crime}"
                
                if ENABLE_USM_DEBUG and r.get("was_sharpened"):
                    label = "[SHARP] " + label
            else:
                label = "Unknown"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            ly = y - 10 if y > 35 else y + h + 20
            cv2.rectangle(frame, (x, ly - th - 5), (x + tw + 8, ly + 4), box_color, -1)
            cv2.putText(frame, label, (x + 4, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_BLACK, 1)

    def _draw_hud(self, frame, face_results, weapon_detections):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 42), (12, 15, 20), -1)
        cv2.putText(frame, "AI FACE + WEAPON RECOGNITION",
                    (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.50, COLOR_ORANGE, 1)
        cv2.putText(frame, f"FPS:{self.stats['fps']:.0f}",
                    (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_GREEN, 1)
        cv2.putText(frame, f"Faces:{len(face_results)}",
                    (90, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_WHITE, 1)
        wc = len(weapon_detections)
        wcolor = COLOR_RED if wc > 0 else COLOR_WHITE
        cv2.putText(frame, f"Weapons:{wc}",
                    (170, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, wcolor, 1)
        cv2.putText(frame, "SFACE",
                    (270, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_YELLOW, 1)
        ts = datetime.now().strftime("%H:%M:%S")
        (tw, _), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.putText(frame, ts, (w - tw - 8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_WHITE, 1)

    # ── Alert handling ────────────────────────────────────────────────────────

    def _handle_face_alert(self, result, frame):
        criminal = result.get("criminal")
        if not criminal:
            return
        cid = criminal["id"]
        now = time.time()
        if now - self._alert_log.get(cid, 0) < ALERT_COOLDOWN_SECONDS:
            return
        self._alert_log[cid] = now

        snapshot_path = None
        if SNAPSHOT_ON_DETECTION:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_path = f"{CAPTURED_DIR}/criminal_{cid}_{ts}.jpg"
            cv2.imwrite(snapshot_path, frame)

        self.db.log_detection(
            criminal_id=cid,
            detected_name=result["name"],
            confidence=result["confidence"],
            snapshot_path=snapshot_path,
            camera_id=self.stats.get("source", "webcam_0"),
        )

        # Compute display confidence % (Fix 4)
        pct = self.engine.remap_confidence(result["confidence"]) * 100
        pct = round(max(0.0, min(100.0, pct)), 1)

        snap_url = None
        if snapshot_path:
            import os
            rel = os.path.relpath(snapshot_path, "data").replace("\\", "/")
            snap_url = f"/images/{rel}"

        self.socketio.emit("alert:face_detected", {
            "name":         result["name"],
            "confidence":   pct,
            "crime_type":   criminal.get("crime_type", "N/A"),
            "status":       criminal.get("status", "N/A"),
            "timestamp":    datetime.now().strftime("%H:%M:%S"),
            "snapshot_url": snap_url,
        })
        logger.warning(f"[PIPELINE] [!] Criminal detected: {result['name']} ({pct}%)")

    def _handle_weapon_alert(self, detections, frame):
        now = time.time()
        types = sorted({d.get("type", "weapon") for d in detections})
        should_alert = any(
            now - self._weapon_alert_log.get(wt, 0) >= WEAPON_ALERT_COOLDOWN_SECONDS
            for wt in types
        )
        if not should_alert:
            return
        for wt in types:
            self._weapon_alert_log[wt] = now

        max_conf = max(float(d.get("confidence", 0)) for d in detections)
        snapshot_path = None
        if WEAPON_SNAPSHOT_ON_DETECTION:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_path = f"{CAPTURED_DIR}/weapon_{ts}.jpg"
            cv2.imwrite(snapshot_path, frame)

        self.db.log_weapon_detection(
            weapon_types=", ".join(types),
            max_confidence=max_conf,
            snapshot_path=snapshot_path,
            camera_id=self.stats.get("source", "webcam_0"),
        )

        snap_url = None
        if snapshot_path:
            import os
            rel = os.path.relpath(snapshot_path, "data").replace("\\", "/")
            snap_url = f"/images/{rel}"

        threat = max(
            (d.get("threat_level", "UNKNOWN") for d in detections),
            key=lambda t: {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1}.get(t, 0)
        )

        self.socketio.emit("alert:weapon_detected", {
            "weapon_types": ", ".join(types),
            "threat_level": threat,
            "confidence":   round(max_conf * 100, 1),
            "timestamp":    datetime.now().strftime("%H:%M:%S"),
            "snapshot_url": snap_url,
        })
        logger.warning(f"[PIPELINE] [!] Weapon detected: {types} ({max_conf:.2f})")

    # ── Offline placeholder ───────────────────────────────────────────────────

    def _make_offline_placeholder(self, is_native=False) -> bytes:
        img = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        img[:] = (15, 18, 25)
        text1 = "NATIVE FEED ACTIVE" if is_native else "FEED OFFLINE"
        text2 = "Check host computer screen" if is_native else "Start a live feed or upload a video"
        cx = FRAME_WIDTH // 2
        cy = FRAME_HEIGHT // 2
        (tw, _), _ = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        color1 = (180, 100, 60) if is_native else (60, 80, 100)
        cv2.putText(img, text1, (cx - tw//2, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color1, 2)
        (tw2, _), _ = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(img, text2, (cx - tw2//2, cy + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (45, 60, 75), 1)
        _, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 50])
        return jpeg.tobytes()
