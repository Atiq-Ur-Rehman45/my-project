"""
mode_manager.py
Focus Mode Toggle Manager for switching between detection modes.

Modes:
    combined    — Both face + weapon detection run simultaneously
    face_only   — Weapon detection OFF, 100% CPU to face recognition
    weapon_only — Face recognition OFF, 100% CPU to weapon detection

This manager coordinates the start/stop of the weapon detector and
the enable/disable of face recognition within the video pipeline.
"""

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class FocusModeManager:
    """
    Manages switching between Combined, Face-Only, and Weapon-Only modes.

    Coordinates with the VideoPipeline to start/stop weapon detection
    and enable/disable face recognition without restarting the camera.

    Thread-safe: mode switches are serialized via a lock.
    """

    MODE_COMBINED = "combined"
    MODE_FACE_ONLY = "face_only"
    MODE_WEAPON_ONLY = "weapon_only"

    VALID_MODES = {MODE_COMBINED, MODE_FACE_ONLY, MODE_WEAPON_ONLY}

    # Display labels and banner colors (BGR for OpenCV)
    MODE_LABELS = {
        MODE_COMBINED: "COMBINED MODE",
        MODE_FACE_ONLY: "FACE ONLY",
        MODE_WEAPON_ONLY: "WEAPON ONLY",
    }

    MODE_COLORS = {
        MODE_COMBINED: (0, 180, 0),       # GREEN
        MODE_FACE_ONLY: (220, 100, 0),    # BLUE
        MODE_WEAPON_ONLY: (0, 0, 220),    # RED
    }

    def __init__(self, initial_mode: str = MODE_COMBINED, socketio=None):
        """
        Args:
            initial_mode: Starting detection mode.
            socketio:     Flask-SocketIO instance for emitting mode change events.
        """
        if initial_mode not in self.VALID_MODES:
            initial_mode = self.MODE_COMBINED

        self._mode = initial_mode
        self._lock = threading.Lock()
        self._socketio = socketio
        self._switching = False  # True during mode transition

        logger.info(f"[MODE] Initialized: {self._mode}")

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def current_mode(self) -> str:
        """Current detection mode string."""
        return self._mode

    @property
    def is_switching(self) -> bool:
        """True if a mode switch is currently in progress."""
        return self._switching

    @property
    def label(self) -> str:
        """Human-readable mode label for UI display."""
        return self.MODE_LABELS.get(self._mode, "UNKNOWN")

    @property
    def banner_color(self) -> tuple:
        """BGR color for the mode banner."""
        return self.MODE_COLORS.get(self._mode, (128, 128, 128))

    @property
    def face_enabled(self) -> bool:
        """Whether face recognition should run in the current mode."""
        return self._mode in (self.MODE_COMBINED, self.MODE_FACE_ONLY)

    @property
    def weapon_enabled(self) -> bool:
        """Whether weapon detection should run in the current mode."""
        return self._mode in (self.MODE_COMBINED, self.MODE_WEAPON_ONLY)

    # ── Mode Switching ────────────────────────────────────────────────────────

    def switch_mode(self, new_mode: str) -> dict:
        """
        Switch to a new detection mode.

        Thread-safe: only one switch can happen at a time.
        Emits WebSocket events for UI notification.

        Args:
            new_mode: Target mode ("combined", "face_only", "weapon_only").

        Returns:
            Dict with switch result: {"success", "mode", "message"}.
        """
        if new_mode not in self.VALID_MODES:
            return {
                "success": False,
                "mode": self._mode,
                "message": f"Invalid mode: {new_mode}. Valid: {self.VALID_MODES}",
            }

        with self._lock:
            if new_mode == self._mode:
                return {
                    "success": True,
                    "mode": self._mode,
                    "message": f"Already in {new_mode} mode.",
                }

            old_mode = self._mode
            self._switching = True

            # Emit "switching" event so UI can show loading indicator
            if self._socketio:
                self._socketio.emit("mode:switching", {
                    "from": old_mode,
                    "to": new_mode,
                })

            logger.info(f"[MODE] Switching: {old_mode} -> {new_mode}")

            # Apply the new mode
            self._mode = new_mode
            self._switching = False

            # Emit "changed" event with new state
            if self._socketio:
                self._socketio.emit("mode:changed", {
                    "mode": new_mode,
                    "label": self.label,
                    "face_enabled": self.face_enabled,
                    "weapon_enabled": self.weapon_enabled,
                })

            logger.info(f"[MODE] Now active: {new_mode} "
                        f"[face={'ON' if self.face_enabled else 'OFF'}, "
                        f"weapon={'ON' if self.weapon_enabled else 'OFF'}]")

            return {
                "success": True,
                "mode": new_mode,
                "message": f"Switched to {self.label}.",
                "face_enabled": self.face_enabled,
                "weapon_enabled": self.weapon_enabled,
            }

    def handle_keypress(self, key: int) -> Optional[str]:
        """
        Handle keyboard shortcuts for mode switching.

        Keys:
            'c' or 'C' → Combined Mode
            'f' or 'F' → Face-Only Mode
            'w' or 'W' → Weapon-Only Mode

        Args:
            key: OpenCV waitKey() return value.

        Returns:
            New mode string if switched, None if key was not a mode shortcut.
        """
        key_char = chr(key & 0xFF).lower() if key >= 0 else ""

        mode_map = {
            'c': self.MODE_COMBINED,
            'f': self.MODE_FACE_ONLY,
            'w': self.MODE_WEAPON_ONLY,
        }

        if key_char in mode_map:
            target = mode_map[key_char]
            if target != self._mode:
                result = self.switch_mode(target)
                if result["success"]:
                    return target
        return None

    # ── Status Info ───────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Full status dict for API responses."""
        return {
            "mode": self._mode,
            "label": self.label,
            "face_enabled": self.face_enabled,
            "weapon_enabled": self.weapon_enabled,
            "is_switching": self._switching,
        }
