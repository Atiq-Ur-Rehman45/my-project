import cv2
import numpy as np
import threading
import time
import logging
from config import (
    FRAME_WIDTH, FRAME_HEIGHT, 
    CAMERA_FPS_TARGET, CAMERA_AUTOFOCUS, CAMERA_BUFFER_SIZE
)

logger = logging.getLogger(__name__)

def _resize_preserve_aspect(img, target_width, target_height):
    h, w = img.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    top = (target_height - new_h) // 2
    bottom = target_height - new_h - top
    left = (target_width - new_w) // 2
    right = target_width - new_w - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

class LatestFrameCamera:
    """Continuously reads from the camera and exposes only the newest frame."""

    def __init__(self, camera_index):
        self.camera_index = camera_index
        self.cap = None
        self._lock = threading.Lock()
        self._frame = None
        self._running = False
        self._thread = None
        self._frame_count = 0

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS_TARGET)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, int(CAMERA_AUTOFOCUS))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)

        try:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        except Exception:
            pass

        if not self.cap.isOpened():
            raise RuntimeError(f"[CAMERA] Cannot open camera index {self.camera_index}")

        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        return self

    def _reader_loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            with self._lock:
                self._frame = frame
                self._frame_count += 1

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def frame_count(self):
        with self._lock:
            return self._frame_count

    def stop(self):
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=1.0)

        if self.cap is not None:
            self.cap.release()

class DirectCamera:
    """Synchronous camera wrapper with the same interface as LatestFrameCamera."""

    def __init__(self, camera_index):
        self.camera_index = camera_index
        self.cap = None
        self._frame_count = 0

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS_TARGET)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, int(CAMERA_AUTOFOCUS))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)

        try:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        except Exception:
            pass

        if not self.cap.isOpened():
            raise RuntimeError(f"[CAMERA] Cannot open camera index {self.camera_index}")

        return self

    def read(self):
        ret, frame = self.cap.read()
        if ret:
            self._frame_count += 1
        return ret, frame

    def frame_count(self):
        return self._frame_count

    def stop(self):
        if self.cap is not None:
            self.cap.release()

class VideoFileCamera:
    """Video file reader with the same interface as camera classes."""

    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = None
        self._frame_count = 0

    def start(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"[CAMERA] Cannot open video file: {self.video_path}")
        return self

    def read(self):
        ret, frame = self.cap.read()
        if ret:
            frame = _resize_preserve_aspect(frame, FRAME_WIDTH, FRAME_HEIGHT)
            self._frame_count += 1
        return ret, frame

    def frame_count(self):
        return self._frame_count

    def stop(self):
        if self.cap is not None:
            self.cap.release()
