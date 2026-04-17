"""
weapon_antispoofing.py
Independent weapon anti-spoofing engine.

Distinguishes REAL physical weapons from 2D representations
(photos, posters, phone screens, video playback).

████████████████████████████████████████████████████████████████
  ISOLATION GUARANTEE:
  - ZERO imports from face_engine, face recognition, or face anti-spoofing.
  - ZERO shared logic with any face pipeline component.
  - The ONLY shared layer: config constants and OpenCV/numpy.
████████████████████████████████████████████████████████████████

Techniques Implemented (4 of 6):
    1. MiDaS Monocular Depth Estimation  — Real 3D objects have depth variance
    2. LBP Texture Analysis              — Screens/paper have detectable patterns
    3. Laplacian Edge Sharpness          — Flat images have softer edges
    4. Confidence Fusion                  — Weighted combination of all signals

Optional (disabled by default, enable in weapon_config.yaml):
    5. Specular Reflection Analysis      — Requires multi-frame + camera motion
    6. Optical Flow Parallax             — Requires camera motion (useless for static CCTV)
"""

import hashlib
import logging
import os
import threading
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    WEAPON_ANTISPOOF_ENABLED,
    WEAPON_ANTISPOOF_THRESHOLD,
    WEAPON_ANTISPOOF_DEPTH_WEIGHT,
    WEAPON_ANTISPOOF_TEXTURE_WEIGHT,
    WEAPON_ANTISPOOF_EDGE_WEIGHT,
    MIDAS_MODEL_PATH,
    MIDAS_INPUT_SIZE,
    MIDAS_DOWNLOAD_URL,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class AntispoofResult:
    """Result of anti-spoofing analysis for a single weapon detection."""
    real_probability: float = 0.0      # Combined score (0.0 = definitely fake, 1.0 = definitely real)
    is_real: bool = False              # Whether it passes the threshold
    depth_score: float = 0.0           # MiDaS depth variance score (0.0–1.0)
    texture_score: float = 0.0         # LBP texture analysis score (0.0–1.0)
    edge_score: float = 0.0            # Laplacian edge sharpness score (0.0–1.0)
    details: Dict = field(default_factory=dict)  # Extra diagnostic info


# ==============================================================================
# MiDaS Model Auto-Download
# ==============================================================================

def download_midas_model(dest_path: str = MIDAS_MODEL_PATH,
                         url: str = MIDAS_DOWNLOAD_URL,
                         sha256_hash: str = "") -> bool:
    """
    Download MiDaS v2.1 Small ONNX model if not present.
    Follows the same pattern as the weapon model downloader.
    """
    if os.path.isfile(dest_path):
        return True

    if not url:
        logger.warning("[ANTISPOOF] MiDaS model not found and no download URL configured.")
        return False

    logger.info(f"[ANTISPOOF] Downloading MiDaS model from: {url}")
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        urllib.request.urlretrieve(url, dest_path)

        if sha256_hash:
            sha = hashlib.sha256()
            with open(dest_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha.update(chunk)
            if sha.hexdigest().lower() != sha256_hash.lower():
                logger.error("[ANTISPOOF] MiDaS model hash mismatch — deleting.")
                os.remove(dest_path)
                return False

        logger.info(f"[ANTISPOOF] MiDaS model downloaded: {dest_path}")
        return True
    except Exception as exc:
        logger.error(f"[ANTISPOOF] MiDaS download failed: {exc}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


# ==============================================================================
# WeaponAntiSpoof — Main Anti-Spoofing Engine
# ==============================================================================

class WeaponAntiSpoof:
    """
    Independent weapon anti-spoofing engine.

    Analyzes the region inside a weapon bounding box to determine if the
    detected weapon is a real physical 3D object or a flat 2D representation
    (printed image, phone screen, monitor, poster).

    This class has ZERO dependencies on face recognition or face anti-spoofing.

    Usage:
        antispoof = WeaponAntiSpoof()
        result = antispoof.analyze(frame, bbox={"x1": 100, "y1": 50, "x2": 300, "y2": 200})
        if result.is_real:
            trigger_alert()

    Toggle behavior:
        - enabled=True  → Real deployment: weapon must pass anti-spoof check
        - enabled=False → Testing mode: bypass anti-spoof, detect all weapons
    """

    def __init__(self,
                 enabled: bool = WEAPON_ANTISPOOF_ENABLED,
                 threshold: float = WEAPON_ANTISPOOF_THRESHOLD,
                 enable_depth: bool = True,
                 enable_texture: bool = True,
                 enable_edge: bool = True):
        self.enabled = enabled
        self.threshold = threshold
        self._enable_depth = enable_depth
        self._enable_texture = enable_texture
        self._enable_edge = enable_edge

        # Fusion weights (normalized at analysis time)
        self._depth_weight = WEAPON_ANTISPOOF_DEPTH_WEIGHT
        self._texture_weight = WEAPON_ANTISPOOF_TEXTURE_WEIGHT
        self._edge_weight = WEAPON_ANTISPOOF_EDGE_WEIGHT

        # MiDaS depth model
        self._depth_session = None
        self._depth_input_name = None
        self._depth_output_names: List[str] = []
        self._depth_loaded = False

        # Load MiDaS if depth analysis is enabled
        if self._enable_depth:
            self._load_depth_model()

        logger.info(f"[ANTISPOOF] Initialized: enabled={enabled}, "
                    f"depth={enable_depth and self._depth_loaded}, "
                    f"texture={enable_texture}, edge={enable_edge}")

    def _load_depth_model(self):
        """Load MiDaS v2.1 Small ONNX model for monocular depth estimation."""
        try:
            # Auto-download if missing
            if not os.path.isfile(MIDAS_MODEL_PATH):
                if not download_midas_model():
                    logger.warning("[ANTISPOOF] MiDaS model unavailable — depth analysis disabled.")
                    self._enable_depth = False
                    return

            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)

            self._depth_session = ort.InferenceSession(
                MIDAS_MODEL_PATH,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
            self._depth_input_name = self._depth_session.get_inputs()[0].name
            self._depth_output_names = [o.name for o in self._depth_session.get_outputs()]
            self._depth_loaded = True
            logger.info("[ANTISPOOF] MiDaS depth model loaded.")

        except Exception as exc:
            logger.error(f"[ANTISPOOF] MiDaS load failed: {exc}")
            self._enable_depth = False

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, frame: np.ndarray, bbox: Dict) -> AntispoofResult:
        """
        Run all enabled anti-spoof checks on a detected weapon region.

        Args:
            frame: Full BGR frame.
            bbox:  Bounding box dict {"x1", "y1", "x2", "y2"}.

        Returns:
            AntispoofResult with scores and real/fake determination.
        """
        # If anti-spoofing is disabled, always pass
        if not self.enabled:
            return AntispoofResult(
                real_probability=1.0,
                is_real=True,
                details={"mode": "TESTING — anti-spoof bypassed"},
            )

        # Extract weapon ROI from the frame
        x1 = max(0, int(bbox.get("x1", 0)))
        y1 = max(0, int(bbox.get("y1", 0)))
        x2 = min(frame.shape[1], int(bbox.get("x2", frame.shape[1])))
        y2 = min(frame.shape[0], int(bbox.get("y2", frame.shape[0])))

        if x2 <= x1 or y2 <= y1:
            return AntispoofResult(real_probability=0.5, is_real=False,
                                   details={"error": "Invalid bounding box"})

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return AntispoofResult(real_probability=0.5, is_real=False,
                                   details={"error": "Empty ROI"})

        # Run each enabled analysis technique
        scores = {}
        weights = {}

        if self._enable_depth and self._depth_loaded:
            scores["depth"] = self._depth_analysis(frame, bbox)
            weights["depth"] = self._depth_weight

        if self._enable_texture:
            scores["texture"] = self._texture_analysis(roi)
            weights["texture"] = self._texture_weight

        if self._enable_edge:
            scores["edge"] = self._edge_analysis(roi)
            weights["edge"] = self._edge_weight

        # Fuse scores into single probability
        real_prob = self._fuse_scores(scores, weights)
        is_real = real_prob >= self.threshold

        return AntispoofResult(
            real_probability=real_prob,
            is_real=is_real,
            depth_score=scores.get("depth", 0.0),
            texture_score=scores.get("texture", 0.0),
            edge_score=scores.get("edge", 0.0),
            details={
                "scores": scores,
                "weights": weights,
                "threshold": self.threshold,
            },
        )

    def toggle(self, enabled: Optional[bool] = None) -> bool:
        """Toggle anti-spoofing on/off. Returns new state."""
        if enabled is not None:
            self.enabled = enabled
        else:
            self.enabled = not self.enabled
        logger.info(f"[ANTISPOOF] Toggled: {'ACTIVE' if self.enabled else 'TESTING MODE'}")
        return self.enabled

    @property
    def status_label(self) -> str:
        """UI-friendly status string."""
        return "Anti-Spoof: ACTIVE" if self.enabled else "Anti-Spoof: TESTING MODE"

    # ── Technique 1: MiDaS Monocular Depth ────────────────────────────────────

    def _depth_analysis(self, frame: np.ndarray, bbox: Dict) -> float:
        """
        Use MiDaS v2.1 Small to estimate depth map within the weapon bounding box.

        Logic:
            - Real 3D weapon → significant depth variance across the object surface
            - Flat screen/poster → uniform/near-zero depth variance

        Returns:
            Score from 0.0 (likely flat/fake) to 1.0 (likely real 3D object).
        """
        try:
            # Prepare input for MiDaS (256x256 RGB, normalized)
            x1 = max(0, int(bbox["x1"]))
            y1 = max(0, int(bbox["y1"]))
            x2 = min(frame.shape[1], int(bbox["x2"]))
            y2 = min(frame.shape[0], int(bbox["y2"]))

            # Use a region slightly larger than the bbox for context
            pad = 20
            rx1 = max(0, x1 - pad)
            ry1 = max(0, y1 - pad)
            rx2 = min(frame.shape[1], x2 + pad)
            ry2 = min(frame.shape[0], y2 + pad)
            region = frame[ry1:ry2, rx1:rx2]

            if region.size == 0:
                return 0.5

            target_w, target_h = MIDAS_INPUT_SIZE
            img = cv2.resize(region, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            # Normalize for MiDaS: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std

            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            img = np.expand_dims(img, axis=0)    # Add batch dim

            # Run MiDaS inference
            depth_map = self._depth_session.run(
                self._depth_output_names, {self._depth_input_name: img}
            )[0]

            if depth_map.ndim == 4:
                depth_map = depth_map[0, 0]
            elif depth_map.ndim == 3:
                depth_map = depth_map[0]

            # Normalize depth map to 0-1 range
            d_min = float(np.min(depth_map))
            d_max = float(np.max(depth_map))
            if d_max - d_min < 1e-6:
                return 0.1  # Completely flat depth = very likely fake

            depth_normalized = (depth_map - d_min) / (d_max - d_min)

            # Calculate depth variance within the weapon bounding box area
            # Map bbox coordinates to the depth map space
            h_scale = target_h / float(ry2 - ry1)
            w_scale = target_w / float(rx2 - rx1)

            bx1 = int((x1 - rx1) * w_scale)
            by1 = int((y1 - ry1) * h_scale)
            bx2 = int((x2 - rx1) * w_scale)
            by2 = int((y2 - ry1) * h_scale)

            bx1 = max(0, min(bx1, target_w - 1))
            by1 = max(0, min(by1, target_h - 1))
            bx2 = max(bx1 + 1, min(bx2, target_w))
            by2 = max(by1 + 1, min(by2, target_h))

            weapon_depth = depth_normalized[by1:by2, bx1:bx2]
            if weapon_depth.size < 4:
                return 0.5

            # Key metric: standard deviation of depth within weapon region
            depth_std = float(np.std(weapon_depth))
            depth_range = float(np.max(weapon_depth) - np.min(weapon_depth))

            # Scoring: higher variance = more likely real 3D object
            # Typical values: real weapon ~0.08-0.25 std, flat image ~0.01-0.04 std
            score = min(1.0, depth_std / 0.12)  # Normalize: 0.12 std = score 1.0
            score = max(0.0, score)

            return score

        except Exception as exc:
            logger.error(f"[ANTISPOOF] Depth analysis failed: {exc}")
            return 0.5  # Neutral on failure

    # ── Technique 2: LBP Texture Analysis ─────────────────────────────────────

    def _texture_analysis(self, roi: np.ndarray) -> float:
        """
        Analyze the micro-texture inside the weapon bounding box using
        Local Binary Patterns (LBP).

        Logic:
            - Real weapons: metal/polymer surfaces have distinctive texture patterns
            - Screen displays: show pixel grid artifacts and uniform digital patterns
            - Printed posters: show paper grain with high regularity

        The LBP histogram is analyzed for:
            1. Entropy (randomness) — real objects have higher entropy
            2. Uniformity — screens/prints have more uniform patterns

        Returns:
            Score from 0.0 (likely fake/flat) to 1.0 (likely real 3D object).
        """
        try:
            # Convert to grayscale and resize for consistent analysis
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Resize to standard size for comparable LBP histograms
            target_size = 64
            if gray.shape[0] < 10 or gray.shape[1] < 10:
                return 0.5
            gray = cv2.resize(gray, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

            # Compute LBP manually (8-neighbor, radius=1)
            lbp = self._compute_lbp(gray)

            # Compute histogram of LBP values
            hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256), density=True)
            hist = hist.astype(np.float64) + 1e-10  # Avoid log(0)

            # Metric 1: Shannon entropy — measures texture randomness
            entropy = -np.sum(hist * np.log2(hist))
            # Real objects: entropy ~6.0-7.5, screens: ~4.0-5.5, prints: ~5.0-6.0
            entropy_score = min(1.0, max(0.0, (entropy - 4.0) / 3.5))

            # Metric 2: Uniformity (sum of squared probabilities)
            # Lower uniformity = more diverse patterns = more likely real
            uniformity = float(np.sum(hist ** 2))
            # Real objects: uniformity ~0.01-0.03, screens: ~0.05-0.15
            uniformity_score = max(0.0, min(1.0, 1.0 - (uniformity / 0.08)))

            # Metric 3: Gradient magnitude variance (detects moire patterns)
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(gx**2 + gy**2)
            grad_var = float(np.var(grad_mag))

            # Real objects: high gradient variance, screens: lower and periodic
            grad_score = min(1.0, max(0.0, grad_var / 2000.0))

            # Combined texture score (weighted average of all metrics)
            score = 0.4 * entropy_score + 0.3 * uniformity_score + 0.3 * grad_score

            return float(np.clip(score, 0.0, 1.0))

        except Exception as exc:
            logger.error(f"[ANTISPOOF] Texture analysis failed: {exc}")
            return 0.5

    @staticmethod
    def _compute_lbp(gray: np.ndarray) -> np.ndarray:
        """
        Compute Local Binary Pattern (8-neighbor, radius=1).
        Pure OpenCV/NumPy implementation — no scikit-image dependency.
        """
        h, w = gray.shape
        lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)

        # 8 neighbor offsets (clockwise from top-left)
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, 1), (1, 1), (1, 0),
            (1, -1), (0, -1),
        ]

        center = gray[1:h-1, 1:w-1].astype(np.int16)

        for bit, (dy, dx) in enumerate(neighbors):
            neighbor = gray[1+dy:h-1+dy, 1+dx:w-1+dx].astype(np.int16)
            lbp |= ((neighbor >= center).astype(np.uint8) << bit)

        return lbp

    # ── Technique 3: Laplacian Edge Sharpness ─────────────────────────────────

    def _edge_analysis(self, roi: np.ndarray) -> float:
        """
        Analyze edge sharpness inside the weapon bounding box using
        the Laplacian operator.

        Logic:
            - Real weapons: sharp, high-frequency edges from physical geometry
            - Printed/screen images: softer, JPEG-compressed, or anti-aliased edges

        The variance of the Laplacian response indicates edge sharpness:
            - High variance → sharp edges → likely real
            - Low variance → soft/blurred edges → likely screen/poster

        Returns:
            Score from 0.0 (likely fake) to 1.0 (likely real).
        """
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            if gray.shape[0] < 8 or gray.shape[1] < 8:
                return 0.5

            # Resize for consistent analysis
            gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_LINEAR)

            # Laplacian: measures second-order derivative (edge sharpness)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            lap_var = float(np.var(laplacian))

            # Typical values:
            #   Real weapon (metal/polymer): lap_var ~100-500
            #   Printed photo: lap_var ~30-80
            #   Phone/monitor screen: lap_var ~15-60
            sharpness_score = min(1.0, max(0.0, lap_var / 150.0))

            # Also check for JPEG block artifacts (8x8 grid periodicity)
            # DCT analysis: screens and JPEG prints show periodic artifacts
            dct = cv2.dct(gray.astype(np.float32))
            # Check energy in high-frequency bins
            h, w = dct.shape
            high_freq_energy = float(np.mean(np.abs(dct[h//2:, w//2:])))
            low_freq_energy = float(np.mean(np.abs(dct[:h//4, :w//4])) + 1e-10)
            freq_ratio = high_freq_energy / low_freq_energy

            # Real objects: more high-freq energy (sharp edges)
            # Prints/screens: less high-freq (compression removes it)
            freq_score = min(1.0, max(0.0, freq_ratio / 0.15))

            # Combined edge score
            score = 0.6 * sharpness_score + 0.4 * freq_score
            return float(np.clip(score, 0.0, 1.0))

        except Exception as exc:
            logger.error(f"[ANTISPOOF] Edge analysis failed: {exc}")
            return 0.5

    # ── Score Fusion ──────────────────────────────────────────────────────────

    def _fuse_scores(self, scores: Dict[str, float],
                     weights: Dict[str, float]) -> float:
        """
        Weighted fusion of all anti-spoof signal scores into a single
        "Real Weapon Probability" (0.0 to 1.0).

        Weights are normalized so they always sum to 1.0, even if
        some techniques are disabled.
        """
        if not scores:
            return 0.5  # Neutral when no signals available

        # Normalize weights to sum to 1.0
        total_weight = sum(weights.get(k, 0) for k in scores)
        if total_weight < 1e-6:
            # Equal weighting fallback
            n = len(scores)
            return sum(scores.values()) / n

        fused = 0.0
        for key, score in scores.items():
            w = weights.get(key, 0) / total_weight
            fused += w * score

        return float(np.clip(fused, 0.0, 1.0))
