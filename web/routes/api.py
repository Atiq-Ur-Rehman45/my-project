"""
web/routes/api.py
All REST API endpoints.
Blueprint prefix: /api
"""

import os
import csv
import io
import threading
import logging
from datetime import datetime
from pathlib import Path

from flask import Blueprint, request, jsonify, current_app, send_file

from config import (
    UPLOAD_DIR, ALLOWED_VIDEO_EXT, UPLOAD_MAX_MB,
    CRIMINAL_DB_DIR, CAPTURED_DIR,
)

logger = logging.getLogger(__name__)
api_bp = Blueprint("api", __name__, url_prefix="/api")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _db():
    return current_app.config["DB"]

def _engine():
    return current_app.config["ENGINE"]

def _trainer():
    return current_app.config["TRAINER"]

def _pipeline():
    return current_app.config["VIDEO_PIPELINE"]

def _enrollment():
    return current_app.config["ENROLLMENT_SERVICE"]

def _socketio():
    return current_app.config["SOCKETIO"]

def _ok(data=None, **kwargs):
    payload = {"success": True}
    if data is not None:
        payload["data"] = data
    payload.update(kwargs)
    return jsonify(payload)

def _err(message, code=400):
    return jsonify({"success": False, "error": message}), code

def _criminal_image_urls(criminal):
    """Return list of web-accessible URLs for a criminal's enrollment images."""
    img_dir = criminal.get("image_dir", "")
    if not img_dir or not os.path.isdir(img_dir):
        return []
    urls = []
    for fname in sorted(os.listdir(img_dir)):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")) and "_crop" not in fname:
            rel = os.path.relpath(os.path.join(img_dir, fname), "data").replace("\\", "/")
            urls.append(f"/images/{rel}")
    return urls


# ══════════════════════════════════════════════════════════════════════════════
# System Status
# ══════════════════════════════════════════════════════════════════════════════

@api_bp.route("/status")
def system_status():
    db     = _db()
    engine = _engine()
    criminals = db.list_all_criminals()

    total_images = 0
    for c in criminals:
        idir = c.get("image_dir") or ""
        if idir and os.path.isdir(idir):
            total_images += len([
                f for f in os.listdir(idir)
                if f.lower().endswith((".jpg", ".png")) and "_crop" not in f
            ])

    # Recognition engine is now fixed to SFACE
    return _ok({
        "enrolled":       len(criminals),
        "model_loaded":   engine.model_loaded,
        "engine":         "SFACE",
        "total_images":   total_images,
        "face_logs":      db.get_detection_count(),
        "weapon_logs":    db.get_weapon_detection_count(),
        "feed_status":    _pipeline().status,
        "feed_source":    _pipeline().stats.get("source"),
        "live_stats":     _pipeline().stats,
    })


# ══════════════════════════════════════════════════════════════════════════════
# Criminals CRUD
# ══════════════════════════════════════════════════════════════════════════════

@api_bp.route("/criminals")
def list_criminals():
    db       = _db()
    search   = (request.args.get("search") or "").lower().strip()
    status_f = (request.args.get("status") or "").strip()
    page     = max(1, int(request.args.get("page", 1)))
    limit    = min(100, int(request.args.get("limit", 20)))

    criminals = db.list_all_criminals()

    # Filter
    if search:
        criminals = [
            c for c in criminals
            if search in (c.get("name") or "").lower()
            or search in (c.get("cnic") or "").lower()
        ]
    if status_f:
        criminals = [c for c in criminals if (c.get("status") or "") == status_f]

    total = len(criminals)
    start = (page - 1) * limit
    page_data = criminals[start:start + limit]

    # Attach image count and thumbnail URL
    for c in page_data:
        urls = _criminal_image_urls(c)
        c["image_count"]   = len(urls)
        c["thumbnail_url"] = urls[0] if urls else None

    return _ok(page_data, total=total, page=page, limit=limit)


@api_bp.route("/criminals", methods=["POST"])
def add_criminal():
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    if not name:
        return _err("Name is required")

    cnic       = (body.get("cnic") or "").strip() or None
    crime_type = (body.get("crime_type") or "Unknown").strip()
    status     = (body.get("status") or "Wanted").strip()
    notes      = (body.get("notes") or "").strip() or None

    try:
        cid, label, img_dir = _db().add_criminal(
            name=name, cnic=cnic,
            crime_type=crime_type, status=status, notes=notes,
        )
    except Exception as e:
        logger.exception("add_criminal failed")
        return _err(str(e), 500)

    return _ok({"id": cid, "face_label": label, "image_dir": img_dir}), 201


@api_bp.route("/criminals/<int:cid>")
def get_criminal(cid):
    record = _db().get_criminal_by_id(cid)
    if not record:
        return _err("Not found", 404)
    record["images"] = _criminal_image_urls(record)
    record["image_count"] = len(record["images"])
    return _ok(record)


@api_bp.route("/criminals/<int:cid>", methods=["PUT"])
def update_criminal(cid):
    body = request.get_json(silent=True) or {}
    try:
        _db().update_criminal(cid, **body)
    except Exception as e:
        return _err(str(e), 500)
    return _ok()


@api_bp.route("/criminals/<int:cid>", methods=["DELETE"])
def delete_criminal(cid):
    purge = request.args.get("purge_snapshots", "false").lower() == "true"
    result = _db().delete_criminal(cid, purge_snapshots=purge)
    if not result.get("deleted"):
        return _err("Record not found or delete failed", 404)

    # Re-train in background using passed instances
    eng_i = _engine()
    tr_i = _trainer()
    db_i = _db()

    def _retrain(trainer, engine, db_inst):
        try:
            success = trainer.full_retrain()
            if success:
                label_map = db_inst.get_label_criminal_map()
                engine.update_label_map(label_map)
        except Exception as e:
            logger.error(f"Retrain after delete failed: {e}")

    threading.Thread(target=_retrain, args=(tr_i, eng_i, db_i), daemon=True).start()
    return _ok(result)


@api_bp.route("/criminals/<int:cid>/images")
def criminal_images(cid):
    record = _db().get_criminal_by_id(cid)
    if not record:
        return _err("Not found", 404)
    return _ok(_criminal_image_urls(record))


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

_training_state = {"status": "idle", "message": "", "percent": 0}


@api_bp.route("/training/status")
def training_status():
    return _ok(_training_state)


@api_bp.route("/training/start", methods=["POST"])
def start_training():
    if _training_state["status"] == "training":
        return _err("Training is already in progress")

    sock = _socketio()
    db_i = _db()
    trainer_i = _trainer()
    engine_i = _engine()

    def _run(socketio, db_inst, trainer, engine):
        try:
            _training_state["status"] = "training"
            _training_state["percent"] = 0
            socketio.emit("training:status", _training_state)

            t0 = datetime.now()
            success = trainer.full_retrain()

            label_map = db_inst.get_label_criminal_map()
            engine.update_label_map(label_map)

            elapsed = (datetime.now() - t0).total_seconds()
            _training_state.update({
                "status":  "complete" if success else "failed",
                "message": f"Done in {elapsed:.1f}s" if success else "Training failed",
                "percent": 100,
            })
            socketio.emit("training:complete", {
                "success": success,
                "elapsed_seconds": round(elapsed, 1),
                "persons_trained": len(db_inst.list_all_criminals()),
            })
        except Exception as exc:
            logger.exception("Training failed")
            _training_state.update({"status": "failed", "message": str(exc), "percent": 0})
            socketio.emit("training:complete", {"success": False, "error": str(exc)})

    threading.Thread(target=_run, args=(sock, db_i, trainer_i, engine_i), daemon=True).start()
    return _ok({"message": "Training started"})


# ══════════════════════════════════════════════════════════════════════════════
# Enrollment Control
# ══════════════════════════════════════════════════════════════════════════════

@api_bp.route("/enrollment/start", methods=["POST"])
def start_enrollment():
    body = request.get_json(silent=True) or {}
    cid  = body.get("criminal_id")
    if not cid:
        return _err("criminal_id required")

    record = _db().get_criminal_by_id(int(cid))
    if not record:
        return _err("Criminal not found", 404)

    svc = _enrollment()
    if svc.is_running:
        # Forcibly cancel any stuck or background enrollment session First
        svc.cancel()
        import time
        for _ in range(20):
            if not svc.is_running:
                break
            time.sleep(0.1)
        if svc.is_running:
            return _err("Could not stop previous enrollment session. Please try again.")

    cam_idx = int(body.get("camera_index", 0))
    ok = svc.start(
        label=record["face_label"],
        save_dir=record["image_dir"],
        camera_index=cam_idx,
    )
    if not ok:
        return _err("Could not start enrollment")
    return _ok({"message": "Enrollment started"})


@api_bp.route("/enrollment/cancel", methods=["POST"])
def cancel_enrollment():
    _enrollment().cancel()
    body = request.get_json(silent=True) or {}
    cid = body.get("criminal_id")
    if cid:
        try:
            _db().delete_criminal(cid, purge_snapshots=True)
        except Exception as e:
            logger.error(f"Failed to delete ghost record {cid}: {e}")
            
    return _ok({"message": "Cancellation requested"})


@api_bp.route("/enrollment/status")
def enrollment_status():
    svc = _enrollment()
    return _ok({
        "status":   svc.status,
        "running":  svc.is_running,
        "collected": len(svc.collected_frames),
    })


# ══════════════════════════════════════════════════════════════════════════════
# Live Feed Control
# ══════════════════════════════════════════════════════════════════════════════

@api_bp.route("/feed/start", methods=["POST"])
def start_feed():
    body   = request.get_json(silent=True) or {}
    source = body.get("source", "camera")
    pipe   = _pipeline()

    if source == "camera":
        cam_idx = int(body.get("camera_index", 0))
        pipe.start_live(camera_index=cam_idx)
        _socketio().emit("status:feed_started", {"source": f"camera:{cam_idx}"})
        return _ok({"message": f"Live camera {cam_idx} feed started"})

    elif source == "video":
        path = body.get("path", "")
        if not path or not os.path.isfile(path):
            return _err(f"Video file not found: {path}")
        pipe.start_video(video_path=path)
        _socketio().emit("status:feed_started", {"source": f"file:{path}"})
        return _ok({"message": f"Video feed started: {path}"})

    return _err("Invalid source. Use 'camera' or 'video'.")


@api_bp.route("/feed/stop", methods=["POST"])
def stop_feed():
    _pipeline().stop()
    return _ok({"message": "Feed stopped"})


@api_bp.route("/feed/pause", methods=["POST"])
def pause_feed():
    _pipeline().pause()
    return _ok()


@api_bp.route("/feed/resume", methods=["POST"])
def resume_feed():
    _pipeline().resume()
    return _ok()


@api_bp.route("/feed/snapshot", methods=["POST"])
def snapshot():
    path = _pipeline().take_snapshot()
    if not path:
        return _err("No active feed or frame not available")
    rel = os.path.relpath(path, "data").replace("\\", "/")
    return _ok({"path": path, "url": f"/images/{rel}"})


@api_bp.route("/feed/status")
def feed_status():
    pipe = _pipeline()
    return _ok({
        "status": pipe.status,
        "stats":  pipe.stats,
    })


# ══════════════════════════════════════════════════════════════════════════════
# Video Upload (drag & drop)
# ══════════════════════════════════════════════════════════════════════════════

@api_bp.route("/upload/video", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        return _err("No file part")
    f    = request.files["file"]
    ext  = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXT:
        return _err(f"Unsupported file type: {ext}")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"upload_{ts}{ext}"
    save_path = os.path.join(UPLOAD_DIR, filename)

    # Stream save to avoid loading huge files into memory
    f.save(save_path)
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    if size_mb > UPLOAD_MAX_MB:
        os.remove(save_path)
        return _err(f"File too large ({size_mb:.0f} MB). Max: {UPLOAD_MAX_MB} MB")

    return _ok({"path": save_path, "filename": filename, "size_mb": round(size_mb, 1)}), 201


# ══════════════════════════════════════════════════════════════════════════════
# Detection Logs
# ══════════════════════════════════════════════════════════════════════════════

@api_bp.route("/logs/faces")
def face_logs():
    limit  = min(200, int(request.args.get("limit", 50)))
    offset = int(request.args.get("offset", 0))
    logs   = _db().get_recent_detections(limit=limit + offset)
    logs   = logs[offset:offset + limit]

    # Patch snapshot_url
    for entry in logs:
        sp = entry.get("snapshot_path")
        if sp and os.path.isfile(sp):
            rel = os.path.relpath(sp, "data").replace("\\", "/")
            entry["snapshot_url"] = f"/images/{rel}"
        else:
            entry["snapshot_url"] = None

    return _ok(logs, total=_db().get_detection_count())


@api_bp.route("/logs/weapons")
def weapon_logs():
    limit  = min(200, int(request.args.get("limit", 50)))
    offset = int(request.args.get("offset", 0))
    logs   = _db().get_recent_weapon_detections(limit=limit + offset)
    logs   = logs[offset:offset + limit]

    for entry in logs:
        sp = entry.get("snapshot_path")
        if sp and os.path.isfile(sp):
            rel = os.path.relpath(sp, "data").replace("\\", "/")
            entry["snapshot_url"] = f"/images/{rel}"
        else:
            entry["snapshot_url"] = None

    return _ok(logs, total=_db().get_weapon_detection_count())


@api_bp.route("/logs/export")
def export_logs():
    log_type = request.args.get("type", "faces")
    if log_type == "faces":
        rows    = _db().get_recent_detections(limit=5000)
        headers = ["id", "timestamp", "detected_name", "confidence", "camera_id", "snapshot_path"]
        fname   = "face_detection_logs.csv"
    else:
        rows    = _db().get_recent_weapon_detections(limit=5000)
        headers = ["id", "timestamp", "weapon_types", "max_confidence", "camera_id", "snapshot_path"]
        fname   = "weapon_detection_logs.csv"

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name=fname,
    )

# ══════════════════════════════════════════════════════════════════════════════
# Settings
# ══════════════════════════════════════════════════════════════════════════════

@api_bp.route("/settings", methods=["GET"])
def get_settings():
    from config import DATA_DIR
    settings_file = os.path.join(DATA_DIR, "settings.json")
    if os.path.exists(settings_file):
        import json
        with open(settings_file, "r") as f:
            return _ok(json.load(f))
    return _ok({})

@api_bp.route("/settings", methods=["PUT"])
def update_settings():
    body = request.get_json(silent=True) or {}
    from config import DATA_DIR
    settings_file = os.path.join(DATA_DIR, "settings.json")
    
    import json
    # Merge with existing
    current = {}
    if os.path.exists(settings_file):
        try:
            with open(settings_file, "r") as f:
                current = json.load(f)
        except:
            pass
            
    current.update(body)
    
    # Save back
    try:
        with open(settings_file, "w") as f:
            json.dump(current, f, indent=4)
        return _ok({"message": "Settings saved"})
    except Exception as e:
        return _err(f"Failed to save settings: {e}")

