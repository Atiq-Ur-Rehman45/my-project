"""
==============================================================
  AI Face Recognition System — Training Manager (SFace)
==============================================================
"""

import cv2
import os
import logging
from config import SFACE_DB_PATH

logger = logging.getLogger(__name__)

class TrainingManager:
    """Handles loading training images and building/updating the recognition model."""

    def __init__(self, face_engine, db_manager):
        self.engine = face_engine
        self.db     = db_manager

    def load_training_data_from_disk(self):
        """
        Walk through criminal_db directory and load all saved face images.
        Formats the images into BGR for SFace embedding calculation.
        """
        criminals = self.db.list_all_criminals()
        training_data = []
        total_images  = 0

        for criminal in criminals:
            label    = criminal["face_label"]
            img_dir  = criminal["image_dir"]
            name     = criminal["name"]

            if not img_dir or not os.path.isdir(img_dir):
                logger.warning(f"[TRAINER] No image directory for {name}, skipping.")
                continue

            count = 0
            for fname in sorted(os.listdir(img_dir)):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                # Fix 3: Skip face-crop thumbnails — they lack surrounding context
                # that YuNet needs for accurate landmark alignment during training.
                if "_crop" in fname:
                    continue
                
                img_path = os.path.join(img_dir, fname)
                
                # ── IMAGE LOADING LOGIC ──────────────────────────────
                # Deep Learning needs full 3-channel BGR colour and high resolution
                img = cv2.imread(img_path)
                if img is None:
                    continue

                training_data.append((img, label))
                count += 1

                # Fix 7: Augment with brightness/contrast variants for robustness
                # This effectively triples the training data without new captures.
                bright = cv2.convertScaleAbs(img, alpha=1.10, beta=15)   # +10% brightness
                dark   = cv2.convertScaleAbs(img, alpha=0.90, beta=-15)  # -10% brightness
                training_data.append((bright, label))
                training_data.append((dark, label))
                count += 2
                # ───────────────────────────────────────────────────

            total_images += count
            print(f"  [TRAINER] {name} (label {label}): {count} images/variants loaded.")

        print(f"[TRAINER] Total: {total_images} images, {len(criminals)} persons.")
        return training_data

    def full_retrain(self):
        """
        Load all enrolled face images and completely retrain the model.
        Use after enrolling new criminals or deleting records.
        """
        print(f"\n[TRAINER] Starting full retrain for SFACE mode...")
        data = self.load_training_data_from_disk()

        if not data:
            print("[TRAINER] ⚠ No training data found. Clearing model state...")

            # Clear runtime state.
            self.engine.model_loaded = False
            self.engine.update_label_map({})

            self.engine.embeddings_db = {}
            self.engine._refresh_sface_index()
            if os.path.exists(SFACE_DB_PATH):
                os.remove(SFACE_DB_PATH)

            print("[TRAINER] ✓ Model state cleared (no enrolled persons remain).")
            return False

        success = self.engine.train(data)
        if success:
            # Refresh label map
            label_map = self.db.get_label_criminal_map()
            self.engine.update_label_map(label_map)
            print(f"[TRAINER] ✓ Model trained with {len(data)} samples.")
        return success

    def get_training_summary(self):
        """Print a summary of enrolled data."""
        criminals = self.db.list_all_criminals()
        print("\n" + "═" * 55)
        print("  ENROLLED PERSONS SUMMARY")
        print("═" * 55)
        for c in criminals:
            img_dir = c.get("image_dir", "")
            count = 0
            if img_dir and os.path.isdir(img_dir):
                count = len([f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))])
            print(f"  [{c['face_label']:>3}] {c['name']:<25} | Images: {count:<4} | Status: {c['status']}")
        print("═" * 55 + "\n")
        return criminals