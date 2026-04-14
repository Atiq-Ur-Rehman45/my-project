import os
import time

import cv2
import psutil

from config import CAMERA_FPS_TARGET, CAMERA_INDEX, FRAME_HEIGHT, FRAME_WIDTH, WARMUP_FRAMES
from database import DatabaseManager
from face_engine import FaceEngine
from monitor import DirectCamera, LatestFrameCamera, LiveMonitor


FRAME_TEST_COUNT = 100
RESOLUTION_TEST_COUNT = 20


def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_stats(fps_samples, frame_times):
    if not fps_samples:
        print("   No frames processed")
        return 0.0

    avg_fps = sum(fps_samples) / len(fps_samples)
    avg_time = sum(frame_times) / len(frame_times) * 1000
    slow_frames = sum(1 for fps in fps_samples if fps < 15)

    print(f"   Average FPS:       {avg_fps:.2f}")
    print(f"   Min FPS:           {min(fps_samples):.2f}")
    print(f"   Max FPS:           {max(fps_samples):.2f}")
    print(f"   Avg Frame Time:    {avg_time:.2f} ms")
    print(f"   Frames Processed:  {len(fps_samples)}")
    print(f"   Slow Frames (<15): {slow_frames} ({slow_frames}%)")
    return avg_fps


def create_capture(width, height):
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS_TARGET)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def warmup_capture(cap, count):
    for _ in range(count):
        cap.read()


def benchmark_engine_only(engine):
    print("\n[1] Engine Pipeline Benchmark...")
    print(f"   Using configured resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    cap = create_capture(FRAME_WIDTH, FRAME_HEIGHT)

    if not cap.isOpened():
        print("   ✗ FAIL - Camera not available")
        raise SystemExit(1)

    print(f"   Warming up camera (discarding first {WARMUP_FRAMES} frames)...")
    warmup_capture(cap, WARMUP_FRAMES)

    fps_samples = []
    frame_times = []
    print(f"   Running {FRAME_TEST_COUNT}-frame engine-only test...")
    for index in range(FRAME_TEST_COUNT):
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            print(f"   ⚠ Frame {index} failed to capture")
            break

        engine.recognize_all_faces(frame)

        elapsed = time.time() - start
        frame_times.append(elapsed)
        fps_samples.append(1.0 / elapsed if elapsed > 0 else 0.0)

        if (index + 1) % 25 == 0:
            print(f"   Progress: {index + 1}/{FRAME_TEST_COUNT} frames")

    cap.release()
    print("\n   Results:")
    return print_stats(fps_samples, frame_times)


def benchmark_camera_readers():
    print("\n[2] Camera Reader Benchmark...")
    print("   Measuring direct read wait vs latest-frame async retrieval")

    for name, camera_cls in (("Direct", DirectCamera), ("AsyncLatest", LatestFrameCamera)):
        camera = camera_cls(CAMERA_INDEX).start()
        try:
            print(f"   {name}: warming up {WARMUP_FRAMES} frames...")
            warmup = 0
            while warmup < WARMUP_FRAMES:
                ret, _ = camera.read()
                if ret:
                    warmup += 1
                else:
                    time.sleep(0.005)

            frame_times = []
            for _ in range(FRAME_TEST_COUNT):
                start = time.time()
                ret, _ = camera.read()
                if not ret:
                    time.sleep(0.005)
                    continue
                frame_times.append(time.time() - start)

            if frame_times:
                avg_ms = sum(frame_times) / len(frame_times) * 1000
                print(f"   {name}: avg read latency {avg_ms:.3f} ms over {len(frame_times)} reads")
        finally:
            camera.stop()


def benchmark_live_pipeline(engine, db):
    print("\n[3] Live Pipeline Benchmark...")
    print("   Measuring flip + recognition + HUD/box drawing without display or alert I/O")

    cap = create_capture(FRAME_WIDTH, FRAME_HEIGHT)
    if not cap.isOpened():
        print("   ✗ FAIL - Camera not available")
        raise SystemExit(1)

    warmup_capture(cap, WARMUP_FRAMES)
    monitor = LiveMonitor(engine, db)

    fps_samples = []
    frame_times = []
    for index in range(FRAME_TEST_COUNT):
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            print(f"   ⚠ Frame {index} failed to capture")
            break

        frame = cv2.flip(frame, 1)
        results, _ = engine.recognize_all_faces(frame)

        for result in results:
            monitor._draw_face_box(frame, result)

        monitor._draw_hud(frame, 0.0, len(results), index + 1)

        elapsed = time.time() - start
        frame_times.append(elapsed)
        fps_samples.append(1.0 / elapsed if elapsed > 0 else 0.0)

        if (index + 1) % 25 == 0:
            print(f"   Progress: {index + 1}/{FRAME_TEST_COUNT} frames")

    cap.release()
    print("\n   Results:")
    return print_stats(fps_samples, frame_times)


def benchmark_memory_usage():
    print("\n[4] Memory Usage Test...")
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"   Memory: {memory_mb:.2f} MB")
    if memory_mb < 500:
        print("   ✓ PASS (Under 500 MB)")
    else:
        print("   ⚠ WARNING (High memory)")


def benchmark_model_load_time():
    print("\n[5] Model Load Time Test...")
    start = time.time()
    FaceEngine()
    load_time = time.time() - start
    print(f"   Load time: {load_time:.3f} seconds")
    if load_time < 3:
        print("   ✓ PASS (Under 3 seconds)")
    else:
        print("   ⚠ WARNING (Slow loading)")


def benchmark_resolution_impact(engine):
    print("\n[6] Resolution Impact Test...")
    for width, height, name in ((640, 480, "SD"), (1280, 720, "HD")):
        cap = create_capture(width, height)
        warmup_capture(cap, 5)

        fps_samples = []
        for _ in range(RESOLUTION_TEST_COUNT):
            start = time.time()
            ret, frame = cap.read()
            if ret:
                engine.recognize_all_faces(frame)
                elapsed = time.time() - start
                fps_samples.append(1.0 / elapsed if elapsed > 0 else 0.0)

        cap.release()
        if fps_samples:
            print(f"   {name} ({width}×{height}): {sum(fps_samples) / len(fps_samples):.2f} FPS")


def print_recommendations(engine_fps, live_fps):
    print("\nRECOMMENDATIONS:")
    if live_fps >= 25:
        print("  ✓ Live pipeline target met at current settings")
    elif engine_fps >= 25 and live_fps < 25:
        print("  1. The model has headroom; remaining loss is in capture, rendering, display, or alert I/O")
        print("  2. Run the actual live monitor and compare idle vs active-alert sessions")
        print("  3. Add stage timing around imshow and alert processing if the gap remains")
    else:
        print("  1. The engine path is still the dominant limiter at this resolution")
        print("  2. Re-check the benchmark environment and camera backend")
        print("  3. Compare with your separate raw model-only benchmark before new architectural changes")


def main():
    print_header("PERFORMANCE BENCHMARK (LIVE-PATH AWARE)")

    db = DatabaseManager()
    engine = FaceEngine()

    engine_fps = benchmark_engine_only(engine)
    benchmark_camera_readers()
    live_fps = benchmark_live_pipeline(engine, db)
    benchmark_memory_usage()
    benchmark_model_load_time()
    benchmark_resolution_impact(engine)

    print_header("BENCHMARK COMPLETE")
    print_recommendations(engine_fps, live_fps)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
