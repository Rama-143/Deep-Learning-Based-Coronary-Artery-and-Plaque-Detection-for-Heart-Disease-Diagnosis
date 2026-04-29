import os
import sqlite3
import uuid
import json
from datetime import timedelta
from datetime import datetime

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, send_from_directory, jsonify
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from inference import run_pipeline


# -------------------- APP --------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "app.db")

UPLOAD_DIR = os.path.join(APP_DIR, "uploads")
RUNS_DIR = os.path.join(APP_DIR, "runs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

ALLOWED_EXTS = (".nii", ".nii.gz")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = "my_super_secret_key_123"   # keep constant
app.permanent_session_lifetime = timedelta(days=7)


def allowed_file(filename: str) -> bool:
    f = filename.lower()
    return any(f.endswith(ext) for ext in ALLOWED_EXTS)


def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema():
    """Create tables and ensure required columns exist."""
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user'
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            username TEXT,
            ct_shape TEXT,
            artery_coverage REAL,
            plaque_percent REAL,
            severe_plaque_percent REAL,
            plaque_grade TEXT
        )
    """)

    # If DB existed before and missing plaque_grade, add safely:
    cur.execute("PRAGMA table_info(results)")
    cols = [r["name"] for r in cur.fetchall()]
    if "plaque_grade" not in cols:
        cur.execute("ALTER TABLE results ADD COLUMN plaque_grade TEXT")

    conn.commit()
    conn.close()


def compute_plaque_grade(plaque_percent: float) -> str:
    if plaque_percent < 30:
        return "Mild"
    elif plaque_percent < 70:
        return "Moderate"
    return "High"


# -------------------- HOME --------------------
@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


# -------------------- REGISTER --------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if not username or not password:
            return render_template("register.html", error="Username and password are required.")

        conn = get_db()
        cur = conn.cursor()

        cur.execute("SELECT 1 FROM users WHERE username=?", (username,))
        if cur.fetchone():
            conn.close()
            return render_template("register.html", error="Username already exists.")

        hashed_password = generate_password_hash(password)

        # First user becomes admin
        cur.execute("SELECT COUNT(*) AS c FROM users")
        count = cur.fetchone()["c"]
        role = "admin" if count == 0 else "user"

        cur.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            (username, hashed_password, role)
        )
        conn.commit()
        conn.close()

        return redirect(url_for("login"))

    return render_template("register.html")


# -------------------- LOGIN --------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT id, username, password, role FROM users WHERE username=?", (username,))
        user = cur.fetchone()
        conn.close()

        if user is None:
            return render_template("login.html",
                                   error="User doesn't exist. Please register.",
                                   show_register=True)

        if not check_password_hash(user["password"], password):
            return render_template("login.html",
                                   error="Wrong password. Try again.",
                                   show_register=False)

        session.clear()
        session.permanent = True
        session["user_id"] = user["id"]
        session["user"] = user["username"]
        session["role"] = user["role"] or "user"

        return redirect(url_for("dashboard"))

    return render_template("login.html", show_register=True)


# -------------------- LOGOUT --------------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# -------------------- DASHBOARD --------------------
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))

    username = session.get("user")
    role = session.get("role", "user")

    conn = get_db()
    cur = conn.cursor()

    if role == "admin":
        cur.execute("SELECT * FROM results ORDER BY id DESC")
    else:
        cur.execute("SELECT * FROM results WHERE username=? ORDER BY id DESC", (username,))

    results = cur.fetchall()
    conn.close()

    return render_template("dashboard.html", results=results, user=username, role=role,)


# -------------------- DELETE RESULT (ADMIN ONLY) --------------------
@app.route("/delete_result/<int:result_id>", methods=["POST"])
def delete_result(result_id):
    if "user" not in session:
        return redirect(url_for("login"))

    if session.get("role") != "admin":
        return "Forbidden", 403

    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM results WHERE id=?", (result_id,))
    conn.commit()
    conn.close()

    return redirect(url_for("dashboard"))


# -------------------- PREDICT --------------------
@app.route("/predict", methods=["POST"])
def predict():
    # require login (recommended)
    if "user" not in session:
        return jsonify({"ok": False, "error": "Login required"}), 401

    ct_file = request.files.get("ct_file")
    gt_file = request.files.get("gt_file")  # optional

    if ct_file is None or ct_file.filename.strip() == "":
        return jsonify({"ok": False, "error": "CT file is required."}), 400

    if not allowed_file(ct_file.filename):
        return jsonify({"ok": False, "error": "CT must be .nii or .nii.gz"}), 400

    run_id = uuid.uuid4().hex[:10]
    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    ct_name = secure_filename(ct_file.filename)
    ct_path = os.path.join(UPLOAD_DIR, f"{run_id}__{ct_name}")
    ct_file.save(ct_path)

    gt_path = None
    if gt_file is not None and gt_file.filename.strip() != "":
        if not allowed_file(gt_file.filename):
            return jsonify({"ok": False, "error": "GT must be .nii or .nii.gz"}), 400
        gt_name = secure_filename(gt_file.filename)
        gt_path = os.path.join(UPLOAD_DIR, f"{run_id}__{gt_name}")
        gt_file.save(gt_path)

    # model pipeline
    result = run_pipeline(ct_path=ct_path, gt_path=gt_path, run_dir=run_dir)

    # guarantee plaque_grade exists for UI + saving
    plaque_percent = float(result.get("plaque_percent", 0.0) or 0.0)
    result["plaque_grade"] = compute_plaque_grade(plaque_percent)

    # save result.json for /results/<run_id>
    with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return jsonify({
        "ok": True,
        "run_id": run_id,
        "result_url": url_for("results", run_id=run_id)
    })


# -------------------- RESULTS PAGE --------------------
@app.route("/results/<run_id>", methods=["GET"])
def results(run_id):
    if "user" not in session:
        return redirect(url_for("login"))

    run_dir = os.path.join(RUNS_DIR, run_id)
    result_json = os.path.join(run_dir, "result.json")
    if not os.path.exists(result_json):
        return "Result not found", 404

    with open(result_json, "r", encoding="utf-8") as f:
        result = json.load(f)

    # guarantee plaque_grade
    plaque_percent = float(result.get("plaque_percent", 0.0) or 0.0)
    result["plaque_grade"] = result.get("plaque_grade") or compute_plaque_grade(plaque_percent)

    current_datetime=datetime.now().strftime("%d-%m-%Y  %H: %M: %S")
    return render_template("result.html", result=result, run_id=run_id, generated_at = current_datetime)


# -------------------- SERVE RUN FILES --------------------
@app.route("/runs/<run_id>/<filename>")
def serve_run_file(run_id, filename):
    return send_from_directory(os.path.join(RUNS_DIR, run_id), filename)


# -------------------- SAVE RESULT (STORE INTO app.db) --------------------
@app.route("/save_result", methods=["POST"])
def save_result():
    if "user" not in session:
        return redirect(url_for("login"))

    run_id = request.form.get("run_id", "").strip()
    if not run_id:
        return "Missing run_id", 400

    # Read from result.json (THIS ensures saved results match prediction)
    run_dir = os.path.join(RUNS_DIR, run_id)
    result_json = os.path.join(run_dir, "result.json")
    if not os.path.exists(result_json):
        return "Result JSON not found", 404

    with open(result_json, "r", encoding="utf-8") as f:
        result = json.load(f)

    ct_shape = str(result.get("ct_shape", ""))
    artery_coverage = float(result.get("artery_coverage_percent", 0) or 0)
    plaque_percent = float(result.get("plaque_percent", 0) or 0)
    severe_plaque_percent = float(result.get("severe_plaque_percent", 0) or 0)
    current_datetime=datetime.now().strftime("%d-%m-%Y  %H: %M: %S")

    plaque_grade = result.get("plaque_grade")
    if not plaque_grade:
        plaque_grade = compute_plaque_grade(plaque_percent)

    conn = get_db()
    cur = conn.cursor()

    # prevent duplicate save (same user + same run_id)
    cur.execute("SELECT 1 FROM results WHERE run_id=? AND username=?", (run_id, session["user"]))
    if cur.fetchone():
        conn.close()
        return redirect(url_for("dashboard"))

    cur.execute("""
        INSERT INTO results (
            run_id, username, ct_shape, artery_coverage,
            plaque_percent, severe_plaque_percent, plaque_grade,current_datetime
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id,
        session["user"],
        ct_shape,
        artery_coverage,
        plaque_percent,
        severe_plaque_percent,
        plaque_grade,
        current_datetime
    ))

    conn.commit()
    conn.close()

    return redirect(url_for("dashboard"))


# -------------------- RUN --------------------
if __name__ == "__main__":
    ensure_schema()
    print("USING DB:", DB_PATH)
    app.run(debug=True)