import sqlite3
import os
import json
from datetime import datetime
from config import DB_PATH, CRIMINAL_DB_DIR


class DatabaseManager:
    """Handles all SQLite operations for criminal records & detection logs."""

    def __init__(self):
        self.db_path = DB_PATH
        self._initialize_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Allows dict-like access
        return conn

    def _initialize_db(self):
        """Create tables if they don't already exist."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS criminals (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    name        TEXT NOT NULL,
                    cnic        TEXT UNIQUE,
                    crime_type  TEXT,
                    status      TEXT DEFAULT 'Wanted',
                    notes       TEXT,
                    face_label  INTEGER UNIQUE,  -- numeric label used by LBPH
                    image_dir   TEXT,            -- directory storing face images
                    created_at  TEXT DEFAULT (datetime('now')),
                    updated_at  TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS detection_logs (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    criminal_id   INTEGER REFERENCES criminals(id),
                    detected_name TEXT,
                    confidence    REAL,
                    camera_id     TEXT DEFAULT 'webcam_0',
                    timestamp     TEXT DEFAULT (datetime('now')),
                    snapshot_path TEXT
                );

                CREATE TABLE IF NOT EXISTS face_label_counter (
                    id      INTEGER PRIMARY KEY CHECK (id = 1),
                    counter INTEGER DEFAULT 0
                );

                INSERT OR IGNORE INTO face_label_counter (id, counter) VALUES (1, 0);
            """)
        print("[DB] Database initialized successfully.")

    # ─── Criminal CRUD ────────────────────────────────────────────────────────

    def add_criminal(self, name, cnic=None, crime_type=None, status="Wanted", notes=None):
        """Add a new criminal record and return their assigned face label."""
        label = self._next_face_label()
        image_dir = os.path.join(CRIMINAL_DB_DIR, f"person_{label}_{name.replace(' ', '_')}")
        os.makedirs(image_dir, exist_ok=True)

        with self._get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO criminals (name, cnic, crime_type, status, notes, face_label, image_dir)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (name, cnic, crime_type, status, notes, label, image_dir)
            )
            criminal_id = cursor.lastrowid

        print(f"[DB] Criminal '{name}' added — ID: {criminal_id}, Face Label: {label}")
        return criminal_id, label, image_dir

    def get_criminal_by_label(self, face_label):
        """Return criminal record dict for a given face label."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM criminals WHERE face_label = ?", (face_label,)
            ).fetchone()
        return dict(row) if row else None

    def get_criminal_by_id(self, criminal_id):
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM criminals WHERE id = ?", (criminal_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_all_criminals(self):
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM criminals ORDER BY id ASC").fetchall()
        return [dict(r) for r in rows]

    def delete_criminal(self, criminal_id):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM criminals WHERE id = ?", (criminal_id,))
        print(f"[DB] Criminal ID {criminal_id} deleted.")

    def update_criminal(self, criminal_id, **kwargs):
        """Update fields on a criminal record."""
        allowed = {"name", "cnic", "crime_type", "status", "notes"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return
        updates["updated_at"] = datetime.now().isoformat()
        sets = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [criminal_id]
        with self._get_connection() as conn:
            conn.execute(f"UPDATE criminals SET {sets} WHERE id = ?", values)

    # ─── Detection Logging ────────────────────────────────────────────────────

    def log_detection(self, criminal_id, detected_name, confidence, snapshot_path=None, camera_id="webcam_0"):
        """Log a recognition event."""
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO detection_logs
                   (criminal_id, detected_name, confidence, camera_id, snapshot_path)
                   VALUES (?, ?, ?, ?, ?)""",
                (criminal_id, detected_name, confidence, camera_id, snapshot_path)
            )

    def get_recent_detections(self, limit=20):
        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT dl.*, c.crime_type, c.status
                   FROM detection_logs dl
                   LEFT JOIN criminals c ON dl.criminal_id = c.id
                   ORDER BY dl.timestamp DESC LIMIT ?""",
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_detection_count(self):
        with self._get_connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM detection_logs").fetchone()[0]

    # ─── Label Management ─────────────────────────────────────────────────────

    def _next_face_label(self):
        """Auto-increment the global face label counter and return the next value."""
        with self._get_connection() as conn:
            conn.execute("UPDATE face_label_counter SET counter = counter + 1 WHERE id = 1")
            row = conn.execute("SELECT counter FROM face_label_counter WHERE id = 1").fetchone()
        return row[0]

    def get_label_name_map(self):
        """Return {face_label: name} dict for all criminals."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT face_label, name FROM criminals").fetchall()
        return {r[0]: r[1] for r in rows}

    def get_label_criminal_map(self):
        """Return {face_label: criminal_dict} for all records."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM criminals").fetchall()
        return {r["face_label"]: dict(r) for r in rows}
