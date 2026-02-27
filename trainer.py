"""
==============================================================
  AI Face Recognition System — Training Manager
  Loads images from disk and trains/retrains the LBPH model
==============================================================
"""

import cv2
import os
import numpy as np
import logging
from config import CRIMINAL_DB_DIR, RECOGNITION_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)


class TrainingManager:
    """Handles loading training images and building/updating the recognition model."""

    def __init__(self, face_engine, db_manager):
        self.engine = face_engine
        self.db     = db_manager

    def load_training_data_from_disk(self):
        """
        Walk through criminal_db directory and load all saved face images.
        
        Expects structure:
            criminal_db/
                person_1_John_Doe/
                    face_0001.jpg
                    face_0002.jpg
                    ...
        
        Each criminal's face_label is queried from DB by matching the directory.
        
        Returns list of (face_ndarray, int_label) tuples.
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
                img_path = os.path.join(img_dir, fname)
                img      = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                # Resize to standard size if needed
                img = cv2.resize(img, (100, 100))
                training_data.append((img, label))
                count += 1

            total_images += count
            print(f"  [TRAINER] {name} (label {label}): {count} images loaded.")

        print(f"[TRAINER] Total: {total_images} images, {len(criminals)} persons.")
        return training_data

    def full_retrain(self):
        """
        Load all enrolled face images and completely retrain the model.
        Use after enrolling new criminals or deleting records.
        """
        print("\n[TRAINER] Starting full retrain...")
        data = self.load_training_data_from_disk()

        if not data:
            print("[TRAINER] ✗ No training data found. Enroll at least one person first.")
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
