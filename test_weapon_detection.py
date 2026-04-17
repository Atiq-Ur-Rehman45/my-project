"""
test_weapon_detection.py
Runner script for testing the independent weapon detection module.

Supports testing via:
- Live Webcam:  python test_weapon_detection.py --source 0
- Video File:   python test_weapon_detection.py --source path/to/video.mp4
- Image/Folder: python test_weapon_detection.py --source path/to/images/

Use --mode to test specific modes (combined, weapon_only).
Use --antispoof to force anti-spoofing logic ON.
"""

import argparse
import glob
import os
import time
import cv2
import numpy as np

from config import WEAPON_CONFIG_PATH
from weapon_detector import WeaponDetector, draw_weapon_detections
from weapon_antispoofing import WeaponAntiSpoof


def main():
    parser = argparse.ArgumentParser(description="Test Weapon Detection Module")
    parser.add_argument("--source", type=str, default="0",
                        help="Camera index, video path, or image directory")
    parser.add_argument("--antispoof", action="store_true",
                        help="Enable anti-spoofing engine")
    parser.add_argument("--config", type=str, default=WEAPON_CONFIG_PATH,
                        help="Path to custom weapon config YAML")
    args = parser.parse_args()

    print("==================================================")
    print("  WEAPON DETECTION & ANTI-SPOOFING TESTBED")
    print("==================================================")

    # Initialize Modules
    detector = WeaponDetector()
    if not detector.model_loaded:
        print("[!] Failed to load weapon model. Check models/ folder.")
        return
    detector.warmup()

    antispoof = None
    if args.antispoof:
        antispoof = WeaponAntiSpoof(enabled=True)
        print("[+] Anti-Spoofing ACTIVE")
    else:
        print("[-] Anti-Spoofing DISABLED (Testing Mode)")

    source = args.source
    
    # ── IMAGE / DIRECTORY TEST ────────────────────────────────────────────────
    if os.path.isdir(source) or source.lower().endswith(('.png', '.jpg', '.jpeg')):
        files = [source] if os.path.isfile(source) else glob.glob(os.path.join(source, "*.[pj][pn][gC]*"))
        print(f"[*] Testing {len(files)} images...")
        
        for fpath in files:
            frame = cv2.imread(fpath)
            if frame is None:
                continue
                
            print(f"\n--- Output for {os.path.basename(fpath)} ---")
            t0 = time.time()
            detections = detector.detect(frame)
            el_det = time.time() - t0
            
            antispoof_results = {}
            if detections and antispoof:
                t1 = time.time()
                for i, det in enumerate(detections):
                    asr = antispoof.analyze(frame, det["bbox"])
                    antispoof_results[i] = {
                        "score": asr.real_probability,
                        "passed": asr.is_real,
                    }
                    print(f"  > AntiSpoof: {asr.is_real} (Score: {asr.real_probability:.3f}) "
                          f"| D: {asr.depth_score:.2f}, T: {asr.texture_score:.2f}, E: {asr.edge_score:.2f}")
                el_as = time.time() - t1
                print(f"  > AS Time: {el_as*1000:.1f}ms")
                
            print(f"  > Det Time: {el_det*1000:.1f}ms | Found: {len(detections)}")
            
            draw_weapon_detections(frame, detections, antispoof_results)
            cv2.imshow("Weapon Test", frame)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        return

    # ── LIVE VIDEO / CAMERA TEST ──────────────────────────────────────────────
    source_idx = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(source_idx)
    
    if not cap.isOpened():
        print(f"[!] Failed to open source: {source}")
        return

    print("[*] Starting video stream... Press 'q' to quit.")
    fps_times = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        t_start = time.time()
        
        # Detection
        detections = detector.detect(frame)
        
        # Anti-spoofing
        antispoof_results = {}
        if detections and antispoof:
            for i, det in enumerate(detections):
                asr = antispoof.analyze(frame, det["bbox"])
                antispoof_results[i] = {
                    "score": asr.real_probability,
                    "passed": asr.is_real,
                }
                
        # Drawing
        draw_weapon_detections(frame, detections, antispoof_results)
        
        # FPS calc
        t_end = time.time()
        fps_times.append(1.0 / max(t_end - t_start, 0.001))
        if len(fps_times) > 30:
            fps_times.pop(0)
        avg_fps = sum(fps_times) / len(fps_times)
        
        # HUD
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if antispoof:
            cv2.putText(frame, "ANTI-SPOOF: ON", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 0), 2)
            
        cv2.imshow("Weapon Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
