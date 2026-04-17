"""
web_app.py
─────────────────────────────────────────────────────────────────────────────
AI Face & Weapon Recognition System — Web Dashboard Entry Point

Run with:
    python web_app.py

Then open:  http://localhost:5000
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import logging
from datetime import datetime

# ── Logging setup (before any other imports) ──────────────────────────────────
from config import LOG_FILE, LOGS_DIR, WEB_HOST, WEB_PORT, WEB_DEBUG, WEB_SECRET_KEY
from config import CRIMINAL_DB_DIR, CAPTURED_DIR, DATA_DIR, UPLOAD_DIR, UPLOAD_MAX_MB

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logging.getLogger("cv2").setLevel(logging.ERROR)
logging.getLogger("werkzeug").setLevel(logging.WARNING)
logger = logging.getLogger("web_app")

# ── Flask & SocketIO ──────────────────────────────────────────────────────────
from flask import Flask, send_from_directory
from flask_socketio import SocketIO

# ── Core ML modules ───────────────────────────────────────────────────────────
from database import DatabaseManager
from face_engine import FaceEngine
from trainer import TrainingManager

# ── Web services ──────────────────────────────────────────────────────────────
from web.services.video_pipeline   import VideoPipeline
from web.services.enrollment_service import EnrollmentService

# ── Blueprints ────────────────────────────────────────────────────────────────
from web.routes.pages  import pages_bp
from web.routes.api    import api_bp
from web.routes.stream import stream_bp


def create_app():
    app = Flask(
        __name__,
        template_folder=os.path.join("web", "templates"),
        static_folder=os.path.join("web", "static"),
    )
    app.config["SECRET_KEY"]        = WEB_SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = UPLOAD_MAX_MB * 1024 * 1024

    # ── SocketIO ──────────────────────────────────────────────────────────────
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode="threading",
        logger=False,
        engineio_logger=False,
    )

    # ── Initialize core ML components ─────────────────────────────────────────
    logger.info("[BOOT] Initializing database...")
    db = DatabaseManager()

    logger.info("[BOOT] Initializing face engine...")
    engine = FaceEngine()

    logger.info("[BOOT] Loading label map...")
    label_map = db.get_label_criminal_map()
    engine.update_label_map(label_map)

    logger.info("[BOOT] Initializing trainer...")
    trainer = TrainingManager(engine, db)

    logger.info("[BOOT] Initializing video pipeline...")
    pipeline = VideoPipeline(engine, db, socketio)

    logger.info("[BOOT] Initializing enrollment service...")
    enrollment_svc = EnrollmentService(engine, socketio)

    # ── Store shared instances in app config ──────────────────────────────────
    app.config["DB"]                 = db
    app.config["ENGINE"]             = engine
    app.config["TRAINER"]            = trainer
    app.config["VIDEO_PIPELINE"]     = pipeline
    app.config["ENROLLMENT_SERVICE"] = enrollment_svc
    app.config["SOCKETIO"]           = socketio

    # ── Register blueprints ───────────────────────────────────────────────────
    app.register_blueprint(pages_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(stream_bp)

    # ── Serve data files (enrollment images + detection snapshots) ────────────
    @app.route("/images/<path:filename>")
    def serve_data_image(filename):
        """
        Serve images from data/ directory.
        URL:  /images/criminal_db/person_1_Ahmed/FRONT_01.jpg
        Maps: data/criminal_db/person_1_Ahmed/FRONT_01.jpg
        """
        return send_from_directory(DATA_DIR, filename)

    # ── WebSocket events ──────────────────────────────────────────────────────
    @socketio.on("connect")
    def on_connect():
        logger.info(f"[WS] Client connected")
        # Send current system state immediately on connect
        socketio.emit("status:init", {
            "enrolled":     len(db.list_all_criminals()),
            "model_loaded": engine.model_loaded,
            "feed_status":  pipeline.status,
        })

    @socketio.on("disconnect")
    def on_disconnect():
        logger.info(f"[WS] Client disconnected")

    @socketio.on("feed:control")
    def on_feed_control(data):
        action = data.get("action")
        if action == "pause":
            pipeline.pause()
        elif action == "resume":
            pipeline.resume()
        elif action == "snapshot":
            path = pipeline.take_snapshot()
            if path:
                import os as _os
                rel = _os.path.relpath(path, DATA_DIR).replace("\\", "/")
                socketio.emit("snapshot:saved", {"url": f"/images/{rel}"})
        elif action == "toggle_native":
            is_active = pipeline.toggle_native_window()
            socketio.emit("status:native_window", {"active": is_active})

    logger.info(f"[BOOT] [OK] Application ready.")
    logger.info(f"[BOOT]   Enrolled: {len(label_map)} person(s)")
    logger.info(f"[BOOT]   Model loaded: {engine.model_loaded}")

    return app, socketio


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 62)
    print("  AI FACE & WEAPON RECOGNITION SYSTEM — Web Dashboard")
    print("=" * 62)
    print(f"  Starting server at http://{WEB_HOST}:{WEB_PORT}")
    print(f"  Open your browser and navigate to: http://localhost:{WEB_PORT}")
    print("=" * 62 + "\n")

    app, socketio = create_app()
    socketio.run(
        app,
        host=WEB_HOST,
        port=WEB_PORT,
        debug=WEB_DEBUG,
        use_reloader=False,    # Disabled to prevent double-loading ML models
        allow_unsafe_werkzeug=True,
    )
