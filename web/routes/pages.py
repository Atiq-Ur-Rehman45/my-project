"""
web/routes/pages.py
Page rendering routes — return HTML templates.
"""
from flask import Blueprint, render_template, current_app
# Recognition engine is fixed to SFACE

pages_bp = Blueprint("pages", __name__)


@pages_bp.route("/")
@pages_bp.route("/dashboard")
def dashboard():
    db     = current_app.config["DB"]
    engine = current_app.config["ENGINE"]
    return render_template("dashboard.html",
        enrolled=len(db.list_all_criminals()),
        model_loaded=engine.model_loaded,
        face_logs=db.get_detection_count(),
        weapon_logs=db.get_weapon_detection_count(),
        engine="SFACE",
        active_page="dashboard",
    )


@pages_bp.route("/enroll")
def enrollment():
    return render_template("enrollment.html", active_page="enroll")


@pages_bp.route("/monitor")
def monitor():
    return render_template("monitor.html", active_page="monitor")


@pages_bp.route("/records")
def records():
    return render_template("records.html", active_page="records")


@pages_bp.route("/logs")
def logs():
    return render_template("logs.html", active_page="logs")


@pages_bp.route("/training")
def training():
    db      = current_app.config["DB"]
    engine  = current_app.config["ENGINE"]
    trainer = current_app.config["TRAINER"]
    criminals = db.list_all_criminals()
    import os
    summary = []
    for c in criminals:
        img_dir = c.get("image_dir", "")
        count = 0
        if img_dir and os.path.isdir(img_dir):
            count = len([
                f for f in os.listdir(img_dir)
                if f.lower().endswith((".jpg", ".png")) and "_crop" not in f
            ])
        summary.append({**c, "image_count": count})
    return render_template("training.html",
        active_page="training",
        criminals=summary,
        model_loaded=engine.model_loaded,
        engine="SFACE",
    )


@pages_bp.route("/settings")
def settings():
    import config as cfg
    return render_template("settings.html",
        active_page="settings",
        cfg=cfg,
    )
