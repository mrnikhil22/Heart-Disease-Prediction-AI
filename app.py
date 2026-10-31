from flask import (
    Flask,
    request,
    render_template,
    send_file,
    jsonify
)
from flask import session, redirect, url_for
from tensorflow.keras.models import load_model

import numpy as np
import io
import os
import pytesseract

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
import sqlite3
import uuid
import datetime
import joblib
import re
import json
import smtplib
from email.message import EmailMessage
import hashlib
import secrets

try:
    import shap
except Exception:
    shap = None

from werkzeug.utils import secure_filename

try:
    import pdfplumber
except Exception:
    pdfplumber = None



try:
    from PIL import Image
except Exception:
    Image = None

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether
)
from reportlab.lib import colors
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)
from reportlab.lib.pagesizes import letter


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "mymodel.keras")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
DATABASE_PATH = os.path.join(BASE_DIR, "heart_ai.db")

# Phase 4 Medical Report Analyzer configuration
MEDICAL_UPLOAD_FOLDER = os.path.join(BASE_DIR, "medical_uploads")
ALLOWED_MEDICAL_EXTENSIONS = {"pdf", "jpg", "jpeg", "png"}
MAX_MEDICAL_FILE_SIZE = 10 * 1024 * 1024
os.makedirs(MEDICAL_UPLOAD_FOLDER, exist_ok=True)
app.config["MEDICAL_UPLOAD_FOLDER"] = MEDICAL_UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_MEDICAL_FILE_SIZE


# ============================================================
# LOAD AI MODEL AND SCALER
# ============================================================

try:
    model = load_model(MODEL_PATH)
    print("AI model loaded successfully.")
except Exception as e:
    model = None
    print("ERROR loading AI model:", e)


try:
    scaler = joblib.load(SCALER_PATH)
    print("Scaler loaded successfully.")
except Exception as e:
    scaler = None
    print("ERROR loading scaler:", e)


# ============================================================
# DATABASE
# ============================================================

def get_db_connection():
    """
    Create and return SQLite database connection.
    """
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def generate_patient_id():
    """Generate a short unique patient ID."""
    return f"PAT-{uuid.uuid4().hex[:6].upper()}"


def initialize_database():
    """
    Create/migrate patient and prediction tables without
    deleting existing prediction history.
    """

    connection = get_db_connection()

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT UNIQUE NOT NULL,
            patient_name TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id TEXT UNIQUE NOT NULL,
            patient_id TEXT,
            patient_name TEXT,
            age REAL NOT NULL,
            gender REAL NOT NULL,
            chestpain REAL NOT NULL,
            resting_bp REAL NOT NULL,
            cholesterol REAL NOT NULL,
            fasting_blood_sugar REAL NOT NULL,
            resting_ecg REAL NOT NULL,
            max_heart_rate REAL NOT NULL,
            exercise_angina REAL NOT NULL,
            oldpeak REAL NOT NULL,
            slope REAL NOT NULL,
            major_vessels REAL NOT NULL,
            thal REAL NOT NULL,
            probability REAL NOT NULL,
            risk_score REAL NOT NULL,
            risk_category TEXT NOT NULL,
            detection_status TEXT NOT NULL,
            explanation TEXT,
            created_at TEXT NOT NULL
        )
        """
    )

    columns = {
        row["name"]
        for row in connection.execute(
            "PRAGMA table_info(predictions)"
        ).fetchall()
    }

    if "patient_id" not in columns:
        connection.execute(
            "ALTER TABLE predictions ADD COLUMN patient_id TEXT"
        )

    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_predictions_patient_id "
        "ON predictions(patient_id)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_predictions_created_at "
        "ON predictions(created_at)"
    )

    connection.commit()

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS medical_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            upload_id TEXT UNIQUE NOT NULL,
            patient_id TEXT,
            original_filename TEXT NOT NULL,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            extraction_method TEXT,
            ocr_used INTEGER DEFAULT 0,
            values_json TEXT,
            analysis_json TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_medical_reports_patient ON medical_reports(patient_id)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_medical_reports_created_at ON medical_reports(created_at)"
    )

    connection.commit()
    connection.close()

    print("Database initialized successfully.")


def get_or_create_patient(patient_id, patient_name):
    """Reuse a patient ID or create a new patient."""

    patient_id = (patient_id or "").strip()
    patient_name = (patient_name or "Anonymous Patient").strip()

    connection = get_db_connection()

    if patient_id:
        row = connection.execute(
            "SELECT patient_id FROM patients WHERE patient_id = ?",
            (patient_id,)
        ).fetchone()

        if row is None:
            connection.close()
            raise ValueError(
                "Patient ID was not found. Leave it blank to create a new patient."
            )

        connection.execute(
            """
            UPDATE patients
            SET patient_name = ?, updated_at = ?
            WHERE patient_id = ?
            """,
            (
                patient_name,
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                patient_id
            )
        )
        connection.commit()
        connection.close()
        return patient_id

    new_id = generate_patient_id()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    connection.execute(
        """
        INSERT INTO patients (
            patient_id, patient_name, created_at, updated_at
        )
        VALUES (?, ?, ?, ?)
        """,
        (new_id, patient_name, now, now)
    )

    connection.commit()
    connection.close()

    return new_id


def get_patient_by_id(patient_id):
    connection = get_db_connection()
    row = connection.execute(
        "SELECT * FROM patients WHERE patient_id = ?",
        (patient_id,)
    ).fetchone()
    connection.close()
    return dict(row) if row else None


def get_patient_history(patient_id, limit=50):
    connection = get_db_connection()
    rows = connection.execute(
        """
        SELECT *
        FROM predictions
        WHERE patient_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (patient_id, limit)
    ).fetchall()
    connection.close()
    return [dict(row) for row in rows]


initialize_database()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value, field_name):
    """
    Safely convert a form value to float.
    """

    if value is None or str(value).strip() == "":
        raise ValueError(f"{field_name} is required.")

    try:
        return float(value)
    except (ValueError, TypeError):
        raise ValueError(f"{field_name} must be a valid number.")


def generate_prediction_id():
    """
    Generate a short unique prediction ID.
    """

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    random_part = uuid.uuid4().hex[:6].upper()

    return f"HD-{timestamp}-{random_part}"


def compute_risk_and_result(prediction_array):
    """
    Convert model output into:
    - risk score
    - risk category
    - probability
    - detection status
    """

    try:
        prob = float(prediction_array[0][0])
    except (IndexError, TypeError, ValueError):
        raise ValueError("Invalid model prediction output.")

    # Keep probability between 0 and 1.
    prob = max(0.0, min(1.0, prob))

    risk_score = round(prob * 100, 1)

    if prob >= 0.75:
        category = "High Risk"
    elif prob >= 0.40:
        category = "Medium Risk"
    else:
        category = "Low Risk"

    # Detection is based on the model's disease probability.
    # This keeps the current model behavior intact.
    if prob >= 0.50:
        detection_status = "Heart Disease Likely Detected"
    else:
        detection_status = "No Strong Heart Disease Indication Detected"

    return (
        risk_score,
        category,
        prob,
        detection_status
    )


def create_explanation(
    age,
    gender,
    chestpain,
    resting_bp,
    cholesterol,
    fasting_blood_sugar,
    resting_ecg,
    max_heart_rate,
    exercise_angina,
    oldpeak,
    slope,
    major_vessels,
    thal,
    category
):
    """
    Create a simple understandable explanation.

    This is NOT SHAP/LIME.
    It is a rule-based explanation based on submitted
    clinical parameters and is clearly treated as an
    explanatory summary rather than a causal ML explanation.
    """

    factors = []

    if age >= 60:
        factors.append("higher age")

    if resting_bp >= 140:
        factors.append("elevated resting blood pressure")

    if cholesterol >= 240:
        factors.append("high cholesterol")

    if fasting_blood_sugar >= 120:
        factors.append("elevated fasting blood sugar")

    if max_heart_rate < 100:
        factors.append("lower maximum heart-rate response")

    if exercise_angina >= 1:
        factors.append("exercise-related chest discomfort")

    if oldpeak >= 2:
        factors.append("higher ST depression value")

    if major_vessels >= 1:
        factors.append("major-vessel related finding")

    if chestpain >= 2:
        factors.append("reported chest-pain pattern")

    if not factors:
        factors.append("the submitted clinical parameter combination")

    if len(factors) == 1:
        factor_text = factors[0]
    elif len(factors) == 2:
        factor_text = f"{factors[0]} and {factors[1]}"
    else:
        factor_text = (
            ", ".join(factors[:-1])
            + ", and "
            + factors[-1]
        )

    return (
        f"The current assessment is classified as {category}. "
        f"The submitted data contains the following potentially "
        f"relevant factors: {factor_text}. "
        f"This is an AI-assisted risk assessment and should not "
        f"be considered a medical diagnosis."
    )


def save_prediction(data):
    """
    Save prediction result and patient parameters into SQLite.
    """

    connection = get_db_connection()

    connection.execute(
        """
        INSERT INTO predictions (
            prediction_id,
            patient_id,
            patient_name,
            age,
            gender,
            chestpain,
            resting_bp,
            cholesterol,
            fasting_blood_sugar,
            resting_ecg,
            max_heart_rate,
            exercise_angina,
            oldpeak,
            slope,
            major_vessels,
            thal,
            probability,
            risk_score,
            risk_category,
            detection_status,
            explanation,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            data["prediction_id"],
            data["patient_id"],
            data["patient_name"],
            data["age"],
            data["gender"],
            data["chestpain"],
            data["resting_bp"],
            data["cholesterol"],
            data["fasting_blood_sugar"],
            data["resting_ecg"],
            data["max_heart_rate"],
            data["exercise_angina"],
            data["oldpeak"],
            data["slope"],
            data["major_vessels"],
            data["thal"],
            data["probability"],
            data["risk_score"],
            data["risk_category"],
            data["detection_status"],
            data["explanation"],
            data["created_at"]
        )
    )

    connection.commit()
    connection.close()


def get_recent_predictions(limit=20):
    """
    Get recent prediction records.
    """

    connection = get_db_connection()

    rows = connection.execute(
        """
        SELECT *
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,)
    ).fetchall()

    connection.close()

    return [dict(row) for row in rows]


def get_prediction_by_id(prediction_id):
    """
    Fetch one prediction by prediction ID.
    """

    connection = get_db_connection()

    row = connection.execute(
        """
        SELECT *
        FROM predictions
        WHERE prediction_id = ?
        """,
        (prediction_id,)
    ).fetchone()

    connection.close()

    if row:
        return dict(row)

    return None


def get_dashboard_statistics():
    """
    Generate dashboard statistics.
    """

    connection = get_db_connection()

    total = connection.execute(
        "SELECT COUNT(*) AS count FROM predictions"
    ).fetchone()["count"]

    high = connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM predictions
        WHERE risk_category = 'High Risk'
        """
    ).fetchone()["count"]

    medium = connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM predictions
        WHERE risk_category = 'Medium Risk'
        """
    ).fetchone()["count"]

    low = connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM predictions
        WHERE risk_category = 'Low Risk'
        """
    ).fetchone()["count"]

    today = datetime.datetime.now().strftime("%Y-%m-%d")

    today_count = connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM predictions
        WHERE substr(created_at, 1, 10) = ?
        """,
        (today,)
    ).fetchone()["count"]

    average_risk = connection.execute(
        """
        SELECT AVG(risk_score) AS average
        FROM predictions
        """
    ).fetchone()["average"]

    connection.close()

    return {
        "total_predictions": total,
        "today_predictions": today_count,
        "high_risk": high,
        "medium_risk": medium,
        "low_risk": low,
        "average_risk": round(
            float(average_risk), 1
        ) if average_risk is not None else 0
    }


# ============================================================
# MAIN PAGE
# ============================================================


# ============================================================
# PHASE 6 — EXPLAINABLE AI (XAI)
# ============================================================

XAI_FEATURE_NAMES = [
    "Age", "Gender", "Chest Pain", "Resting BP", "Cholesterol",
    "Fasting Blood Sugar", "Resting ECG", "Max Heart Rate",
    "Exercise Angina", "Oldpeak", "Slope", "Major Vessels", "Thal"
]

def build_xai_explanation(scaled_input):
    values = np.asarray(scaled_input, dtype=float).reshape(1, -1)
    contributions = []
    method = "Local feature sensitivity"

    if shap is not None:
        try:
            background = np.zeros((1, values.shape[1]), dtype=float)
            def predict_fn(x):
                return np.asarray(model.predict(x, verbose=0)).reshape(-1)
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(values, nsamples=100)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            shap_values = np.asarray(shap_values).reshape(-1)
            method = "SHAP (KernelExplainer)"
            for name, val, contribution in zip(XAI_FEATURE_NAMES, values[0], shap_values):
                contributions.append({"feature": name, "input_value": float(val), "contribution": float(contribution)})
        except Exception as exc:
            print("SHAP explanation unavailable, using local sensitivity:", exc)

    if not contributions:
        try:
            base = float(model.predict(values, verbose=0)[0][0])
            for i, name in enumerate(XAI_FEATURE_NAMES):
                perturbed = values.copy()
                delta = max(abs(float(values[0, i])) * 0.05, 0.05)
                perturbed[0, i] += delta
                changed = float(model.predict(perturbed, verbose=0)[0][0])
                contributions.append({"feature": name, "input_value": float(values[0, i]), "contribution": changed - base})
        except Exception:
            contributions = []

    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    top = []
    for item in contributions[:5]:
        c = item["contribution"]
        top.append({
            "feature": item["feature"],
            "contribution": round(c, 6),
            "direction": "Higher predicted probability" if c > 0 else "Lower predicted probability" if c < 0 else "Neutral"
        })
    return {
        "available": bool(top),
        "method": method,
        "features": top,
        "summary": ("The explanation shows which input features had the strongest local influence on this model prediction. Positive contribution moved the model output upward; negative contribution moved it downward." if top else "Feature-level explanation is currently unavailable."),
    }

@app.route("/")
def index():
    """
    Main page - Dashboard.
    """
    return render_template("dashboard.html")


@app.route("/prediction")
def prediction():
    """
    Prediction page.
    """
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    """
    Dashboard page.
    """
    return render_template("dashboard.html")
# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():
    """
    Existing prediction route.

    The original 13 form fields are preserved.
    """

    try:

        if model is None:
            return (
                "<h2>AI model is not available.</h2>",
                500
            )

        if scaler is None:
            return (
                "<h2>Scaler is not available.</h2>",
                500
            )

        # ----------------------------------------------------
        # Optional patient name
        # ----------------------------------------------------

        patient_name = request.form.get(
            "patient_name",
            request.form.get(
                "name",
                "Anonymous Patient"
            )
        ).strip()

        if not patient_name:
            patient_name = "Anonymous Patient"

        patient_id = request.form.get("patient_id", "").strip()

        # Privacy rule: a PATIENT account can only create a prediction
        # for the Patient ID linked to its own account.
        logged_in_user = current_user()
        if logged_in_user and logged_in_user.get("role") == "PATIENT":
            patient_id = logged_in_user.get("patient_id") or ""

        patient_id = get_or_create_patient(
            patient_id,
            patient_name
        )

        # ----------------------------------------------------
        # Read the original 13 inputs
        # ----------------------------------------------------

        age = safe_float(
            request.form.get("age"),
            "Age"
        )

        gender = safe_float(
            request.form.get("gender"),
            "Gender"
        )

        chestpain = safe_float(
            request.form.get("chestpain"),
            "Chest Pain"
        )

        resting_bp = safe_float(
            request.form.get("restingBP"),
            "Resting Blood Pressure"
        )

        cholesterol = safe_float(
            request.form.get("serumcholestrol"),
            "Cholesterol"
        )

        fasting_blood_sugar = safe_float(
            request.form.get("fastingbloodsugar"),
            "Fasting Blood Sugar"
        )

        resting_ecg = safe_float(
            request.form.get("restingrelectro"),
            "Resting ECG"
        )

        max_heart_rate = safe_float(
            request.form.get("maxheartrate"),
            "Maximum Heart Rate"
        )

        exercise_angina = safe_float(
            request.form.get("exerciseangia"),
            "Exercise Angina"
        )

        oldpeak = safe_float(
            request.form.get("oldpeak"),
            "Oldpeak"
        )

        slope = safe_float(
            request.form.get("slope"),
            "Slope"
        )

        major_vessels = safe_float(
            request.form.get("noofmajorvessels"),
            "Number of Major Vessels"
        )

        thal = safe_float(
            request.form.get("thal"),
            "Thal"
        )

        # ----------------------------------------------------
        # Basic validation
        # ----------------------------------------------------

        if age < 1 or age > 120:
            raise ValueError(
                "Age must be between 1 and 120."
            )

        if resting_bp < 0 or resting_bp > 300:
            raise ValueError(
                "Resting blood pressure value is invalid."
            )

        if cholesterol < 0 or cholesterol > 1000:
            raise ValueError(
                "Cholesterol value is invalid."
            )

        if max_heart_rate < 0 or max_heart_rate > 300:
            raise ValueError(
                "Maximum heart rate value is invalid."
            )

        if oldpeak < -20 or oldpeak > 20:
            raise ValueError(
                "Oldpeak value is invalid."
            )

        # ----------------------------------------------------
        # Create model input
        # ----------------------------------------------------

        inputs = [
            age,
            gender,
            chestpain,
            resting_bp,
            cholesterol,
            fasting_blood_sugar,
            resting_ecg,
            max_heart_rate,
            exercise_angina,
            oldpeak,
            slope,
            major_vessels,
            thal
        ]

        input_array = np.array(
            [inputs],
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Apply saved scaler
        # ----------------------------------------------------

        input_array = scaler.transform(
            input_array
        )

        # ----------------------------------------------------
        # AI prediction
        # ----------------------------------------------------

        prediction = model.predict(
            input_array,
            verbose=0
        )

        (
            risk_score,
            category,
            prob,
            detection_status
        ) = compute_risk_and_result(
            prediction
        )

        # ----------------------------------------------------
        # Explanation
        # ----------------------------------------------------

        explanation = create_explanation(
            age,
            gender,
            chestpain,
            resting_bp,
            cholesterol,
            fasting_blood_sugar,
            resting_ecg,
            max_heart_rate,
            exercise_angina,
            oldpeak,
            slope,
            major_vessels,
            thal,
            category
        )

        # ----------------------------------------------------
        # Create unique prediction ID
        # ----------------------------------------------------

        prediction_id = generate_prediction_id()

        created_at = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # ----------------------------------------------------
        # Save to database
        # ----------------------------------------------------

        prediction_data = {
            "prediction_id": prediction_id,
            "patient_id": patient_id,
            "patient_name": patient_name,
            "age": age,
            "gender": gender,
            "chestpain": chestpain,
            "resting_bp": resting_bp,
            "cholesterol": cholesterol,
            "fasting_blood_sugar": fasting_blood_sugar,
            "resting_ecg": resting_ecg,
            "max_heart_rate": max_heart_rate,
            "exercise_angina": exercise_angina,
            "oldpeak": oldpeak,
            "slope": slope,
            "major_vessels": major_vessels,
            "thal": thal,
            "probability": round(prob, 3),
            "risk_score": risk_score,
            "risk_category": category,
            "detection_status": detection_status,
            "explanation": explanation,
            "created_at": created_at
        }

        save_prediction(prediction_data)

        # ----------------------------------------------------
        # Phase 6 — Explainable AI
        # ----------------------------------------------------

        xai_explanation = build_xai_explanation(input_array)

        # ----------------------------------------------------
        # Render existing result page
        # ----------------------------------------------------

        return render_template(
            "result.html",

            # Existing variables
            risk_score=risk_score,
            category=category,
            probability=round(prob, 3),

            # New variables
            detection_status=detection_status,
            prediction_id=prediction_id,
            patient_id=patient_id,
            patient_name=patient_name,
            explanation=explanation,
            xai_explanation=xai_explanation,

            # Clinical values
            age=age,
            gender=gender,
            chestpain=chestpain,
            restingBP=resting_bp,
            serumcholestrol=cholesterol,
            fastingbloodsugar=fasting_blood_sugar,
            restingrelectro=resting_ecg,
            maxheartrate=max_heart_rate,
            exerciseangia=exercise_angina,
            oldpeak=oldpeak,
            slope=slope,
            noofmajorvessels=major_vessels,
            thal=thal
        )

    except ValueError as e:

        print("VALIDATION ERROR:", e)

        return (
            f"""
            <html>
                <head>
                    <title>Invalid Input</title>
                </head>
                <body>
                    <h2>Invalid Input</h2>
                    <p>{str(e)}</p>
                    <a href="/prediction">Go Back</a>
                </body>
            </html>
            """,
            400
        )

    except Exception as e:

        print("PREDICTION ERROR:", e)

        return (
            """
            <html>
                <head>
                    <title>Prediction Error</title>
                </head>
                <body>
                    <h2>Unable to process prediction.</h2>
                    <p>Please check the entered values and try again.</p>
                    <a href="/prediction">Go Back</a>
                </body>
            </html>
            """,
            500
        )


# ============================================================
# PDF REPORT
# ============================================================


@app.route("/api/xai/<prediction_id>", methods=["GET"])
def api_xai_prediction(prediction_id):
    try:
        prediction = get_prediction_by_id(prediction_id)
        if prediction is None:
            return jsonify({"success": False, "error": "Prediction not found."}), 404
        raw = [prediction.get(k) for k in [
            "age", "gender", "chestpain", "resting_bp", "cholesterol",
            "fasting_blood_sugar", "resting_ecg", "max_heart_rate",
            "exercise_angina", "oldpeak", "slope", "major_vessels", "thal"
        ]]
        scaled = scaler.transform([raw])
        return jsonify({"success": True, "prediction_id": prediction_id, "explanation": build_xai_explanation(scaled)})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500

@app.route("/report", methods=["POST"])
def report():
    """
    Generate the Phase 3 professional medical assessment report.

    The route keeps the existing /report POST workflow and uses the
    stored prediction record whenever a prediction ID is available.
    If the existing result form does not send prediction_id, the latest
    prediction for the supplied patient is used as a compatibility fallback.
    """

    try:

        # ----------------------------------------------------
        # Identify the saved prediction
        # ----------------------------------------------------

        prediction_id = request.form.get(
            "prediction_id",
            ""
        ).strip()

        patient_id = request.form.get(
            "patient_id",
            ""
        ).strip()

        prediction = None

        if prediction_id:
            prediction = get_prediction_by_id(prediction_id)

        # Compatibility with the current result.html form, which
        # already sends patient_id but may not yet send prediction_id.
        if prediction is None and patient_id:
            history = get_patient_history(patient_id, 1)
            if history:
                prediction = history[0]

        # ----------------------------------------------------
        # Use stored prediction data when available
        # ----------------------------------------------------

        if prediction:

            prediction_id = prediction.get(
                "prediction_id",
                prediction_id or "N/A"
            )

            patient_id = prediction.get(
                "patient_id",
                patient_id or "N/A"
            )

            patient_name = prediction.get(
                "patient_name"
            ) or request.form.get(
                "patient_name",
                "Anonymous Patient"
            )

            risk_score = prediction.get(
                "risk_score",
                request.form.get("risk_score", "N/A")
            )

            category = prediction.get(
                "risk_category",
                request.form.get("category", "N/A")
            )

            probability = prediction.get(
                "probability",
                request.form.get("probability", "N/A")
            )

            detection_status = prediction.get(
                "detection_status",
                "Not available"
            )

            explanation = prediction.get(
                "explanation",
                "No explanation available."
            )

        else:

            # Existing form values remain supported even when a
            # saved prediction record cannot be found.
            prediction_id = prediction_id or request.form.get(
                "prediction_id",
                "N/A"
            )

            patient_name = request.form.get(
                "patient_name",
                "Anonymous Patient"
            )

            risk_score = request.form.get(
                "risk_score",
                "N/A"
            )

            category = request.form.get(
                "category",
                "N/A"
            )

            probability = request.form.get(
                "probability",
                "N/A"
            )

            detection_status = request.form.get(
                "detection_status",
                "Not available"
            )

            explanation = request.form.get(
                "explanation",
                "No explanation available."
            )

            patient_id = patient_id or "N/A"

        # ----------------------------------------------------
        # Clinical values
        # ----------------------------------------------------

        clinical_values = {}

        clinical_fields = [
            ("Age", "age", "age"),
            ("Gender", "gender", "gender"),
            ("Chest Pain Type", "chestpain", "chestpain"),
            ("Resting Blood Pressure", "resting_bp", "restingBP"),
            ("Cholesterol", "cholesterol", "serumcholestrol"),
            ("Fasting Blood Sugar", "fasting_blood_sugar", "fastingbloodsugar"),
            ("Resting ECG", "resting_ecg", "restingrelectro"),
            ("Maximum Heart Rate", "max_heart_rate", "maxheartrate"),
            ("Exercise Angina", "exercise_angina", "exerciseangia"),
            ("Oldpeak", "oldpeak", "oldpeak"),
            ("Slope", "slope", "slope"),
            ("Major Vessels", "major_vessels", "noofmajorvessels"),
            ("Thal", "thal", "thal")
        ]

        for label, db_key, form_key in clinical_fields:

            if prediction and prediction.get(db_key) is not None:
                value = prediction.get(db_key)
            else:
                value = request.form.get(
                    form_key,
                    "N/A"
                )

            clinical_values[label] = value

        # ----------------------------------------------------
        # Phase 3 health score
        # ----------------------------------------------------
        # This is a simple risk-derived software score, not a
        # clinical health score and not an independently validated
        # medical measurement.

        try:
            numeric_risk = float(risk_score)
            health_score = max(
                0,
                min(
                    100,
                    round(100 - numeric_risk, 1)
                )
            )
        except (ValueError, TypeError):
            health_score = "N/A"

        # ----------------------------------------------------
        # Key contributing factors
        # ----------------------------------------------------
        # These are rule-based observations from submitted values.
        # They are not SHAP/LIME feature-importance values.

        factors = []

        try:
            age_value = float(clinical_values["Age"])
            if age_value >= 60:
                factors.append("Age is 60 years or above.")
        except (ValueError, TypeError):
            pass

        try:
            bp_value = float(clinical_values["Resting Blood Pressure"])
            if bp_value >= 140:
                factors.append("Resting blood pressure is elevated based on the entered value.")
        except (ValueError, TypeError):
            pass

        try:
            cholesterol_value = float(clinical_values["Cholesterol"])
            if cholesterol_value >= 240:
                factors.append("Cholesterol value is high based on the entered value.")
        except (ValueError, TypeError):
            pass

        try:
            heart_rate_value = float(clinical_values["Maximum Heart Rate"])
            if heart_rate_value < 100:
                factors.append("Maximum heart-rate value entered is below 100 bpm.")
        except (ValueError, TypeError):
            pass

        try:
            oldpeak_value = float(clinical_values["Oldpeak"])
            if oldpeak_value > 1:
                factors.append("Oldpeak value is above 1.0.")
        except (ValueError, TypeError):
            pass

        try:
            exercise_angina_value = float(clinical_values["Exercise Angina"])
            if exercise_angina_value == 1:
                factors.append("Exercise-induced angina is marked as present.")
        except (ValueError, TypeError):
            pass

        try:
            major_vessel_value = float(clinical_values["Major Vessels"])
            if major_vessel_value > 0:
                factors.append("One or more major vessels are recorded in the input.")
        except (ValueError, TypeError):
            pass

        if not factors:
            factors.append(
                "No rule-based contributing factor was flagged from the entered values."
            )

        # ----------------------------------------------------
        # Recommendations
        # ----------------------------------------------------

        if category == "High Risk":
            recommendation_text = (
                "The assessment indicates a higher predicted risk. "
                "Discuss the result and the submitted clinical information "
                "with a qualified healthcare professional for appropriate "
                "clinical evaluation and further testing if required."
            )
        elif category == "Medium Risk":
            recommendation_text = (
                "The assessment indicates a moderate predicted risk. "
                "Consider discussing the result and clinical parameters "
                "with a qualified healthcare professional and maintaining "
                "appropriate heart-healthy habits."
            )
        else:
            recommendation_text = (
                "The assessment indicates a lower predicted risk based on "
                "the submitted information. Continue appropriate healthy "
                "lifestyle practices and routine medical check-ups as advised."
            )

        # ----------------------------------------------------
        # Doctor discussion section
        # ----------------------------------------------------

        doctor_discussion = (
            "Discuss the AI-assisted risk result, model probability, "
            "clinical parameters entered, symptoms, medical history, "
            "current medicines, and whether additional clinical tests "
            "are appropriate."
        )

        # ----------------------------------------------------
        # Create PDF buffer
        # ----------------------------------------------------

        buffer = io.BytesIO()

        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=42,
            leftMargin=42,
            topMargin=42,
            bottomMargin=42
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            name="Phase3ReportTitle",
            parent=styles["Title"],
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#0f766e"),
            spaceAfter=5,
            alignment=1
        )

        subtitle_style = ParagraphStyle(
            name="Phase3ReportSubtitle",
            parent=styles["Normal"],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#475569"),
            spaceAfter=14,
            alignment=1
        )

        section_style = ParagraphStyle(
            name="Phase3Section",
            parent=styles["Heading2"],
            fontSize=13,
            leading=17,
            textColor=colors.HexColor("#0f766e"),
            spaceBefore=10,
            spaceAfter=8
        )

        normal_style = ParagraphStyle(
            name="Phase3Normal",
            parent=styles["Normal"],
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor("#1e293b")
        )

        small_style = ParagraphStyle(
            name="Phase3Small",
            parent=styles["Normal"],
            fontSize=7.5,
            leading=10.5,
            textColor=colors.HexColor("#64748b")
        )

        metric_style = ParagraphStyle(
            name="Phase3Metric",
            parent=styles["Normal"],
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#0f172a"),
            alignment=1
        )

        content = []

        # ----------------------------------------------------
        # Professional header
        # ----------------------------------------------------

        content.append(
            Paragraph(
                "HEART AI",
                title_style
            )
        )

        content.append(
            Paragraph(
                "AI-Driven Heart Disease Prediction, Detection, and Classification System",
                subtitle_style
            )
        )

        current_date = datetime.datetime.now().strftime(
            "%d-%m-%Y %H:%M:%S"
        )

        patient_data = [
            ["Patient Name", str(patient_name)],
            ["Patient ID", str(patient_id)],
            ["Report ID", str(prediction_id)],
            ["Date & Time", current_date]
        ]

        patient_table = Table(
            patient_data,
            colWidths=[130, 330]
        )

        patient_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e6fffb")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#0f766e")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("PADDING", (0, 0), (-1, -1), 7)
            ])
        )

        content.append(patient_table)
        content.append(Spacer(1, 14))

        # ----------------------------------------------------
        # Assessment summary
        # ----------------------------------------------------

        content.append(
            Paragraph("Assessment Summary", section_style)
        )

        result_data = [
            ["Risk Classification", str(category)],
            ["Risk Score", f"{risk_score}%"],
            ["Model Probability", str(probability)],
            ["Detection Status", str(detection_status)],
            ["Health Score", f"{health_score}/100"]
        ]

        result_table = Table(
            result_data,
            colWidths=[220, 240]
        )

        result_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0fdfa")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#0f766e")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("PADDING", (0, 0), (-1, -1), 7)
            ])
        )

        content.append(result_table)
        content.append(Spacer(1, 14))

        content.append(
            Paragraph(
                "Health Score note: this score is a simple software-derived value of 100 minus the model risk score. It is not a clinically validated health measurement.",
                small_style
            )
        )

        # ----------------------------------------------------
        # Clinical parameters
        # ----------------------------------------------------

        content.append(
            Paragraph("Clinical Parameters", section_style)
        )

        clinical_rows = [
            ["Clinical Parameter", "Entered Value"]
        ]

        for label, value in clinical_values.items():
            clinical_rows.append([
                str(label),
                str(value)
            ])

        clinical_table = Table(
            clinical_rows,
            colWidths=[270, 190],
            repeatRows=1
        )

        clinical_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
            ])
        )

        content.append(clinical_table)
        content.append(Spacer(1, 14))

        # ----------------------------------------------------
        # AI-assisted explanation
        # ----------------------------------------------------

        content.append(
            Paragraph("AI-Assisted Explanation", section_style)
        )

        content.append(
            Paragraph(
                str(explanation),
                normal_style
            )
        )

        content.append(Spacer(1, 10))

        # ----------------------------------------------------
        # Key contributing factors
        # ----------------------------------------------------

        content.append(
            Paragraph("Key Contributing Factors", section_style)
        )

        factor_rows = []
        for factor in factors:
            factor_rows.append(["•", factor])

        factor_table = Table(
            factor_rows,
            colWidths=[18, 442]
        )

        factor_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 6)
            ])
        )

        content.append(factor_table)
        content.append(
            Spacer(1, 5)
        )
        content.append(
            Paragraph(
                "These factors are rule-based observations from the entered clinical values. They are not SHAP/LIME feature-importance results.",
                small_style
            )
        )

        # ----------------------------------------------------
        # Recommendations
        # ----------------------------------------------------

        content.append(
            Paragraph("General Recommendations", section_style)
        )

        content.append(
            Paragraph(
                recommendation_text,
                normal_style
            )
        )

        # ----------------------------------------------------
        # Doctor discussion
        # ----------------------------------------------------

        content.append(
            Paragraph("Doctor Discussion", section_style)
        )

        content.append(
            Paragraph(
                doctor_discussion,
                normal_style
            )
        )

        # ----------------------------------------------------
        # Report verification structure
        # ----------------------------------------------------

        content.append(
            Paragraph("Report Verification", section_style)
        )

        verification_text = (
            f"Report ID: <b>{prediction_id}</b><br/>"
            "This identifier can be used as the future verification key for a QR-based report verification module."
        )

        content.append(
            Paragraph(
                verification_text,
                normal_style
            )
        )

        content.append(Spacer(1, 16))

        # ----------------------------------------------------
        # Disclaimer
        # ----------------------------------------------------

        disclaimer = (
            "<b>Medical Disclaimer:</b> This report is generated by an AI-assisted software system for educational and decision-support purposes. "
            "It is not a definitive medical diagnosis and does not replace evaluation by a qualified healthcare professional. "
            "Clinical decisions should be based on appropriate medical examination, history, tests, and professional judgment."
        )

        content.append(
            Paragraph(
                disclaimer,
                small_style
            )
        )

        # ----------------------------------------------------
        # Build PDF
        # ----------------------------------------------------

        doc.build(content)

        buffer.seek(0)

        return send_file(
            buffer,
            as_attachment=True,
            download_name="HeartAI_Professional_Medical_Report.pdf",
            mimetype="application/pdf"
        )

    except Exception as e:

        print("REPORT ERROR:", e)

        return (
            "<h2>Unable to generate report.</h2>"
            "<p>Please try again.</p>",
            500
        )


# ============================================================
# HISTORY PAGE
# ============================================================

@app.route("/history")
def history():
    """Phase 5 history and analytics page."""
    return render_template("history.html")


# ============================================================
# SINGLE PREDICTION API
# ============================================================

@app.route("/api/prediction/<prediction_id>")
def prediction_details(prediction_id):
    """
    Get a single prediction.
    """

    record = get_prediction_by_id(
        prediction_id
    )

    if record is None:

        return jsonify({
            "success": False,
            "message": "Prediction not found."
        }), 404

    return jsonify({
        "success": True,
        "prediction": record
    })



# ============================================================
# PATIENT APIs
# ============================================================

@app.route("/api/patient/<patient_id>")
def patient_details(patient_id):
    patient = get_patient_by_id(patient_id)

    if patient is None:
        return jsonify({
            "success": False,
            "message": "Patient not found."
        }), 404

    history = get_patient_history(patient_id, 50)

    return jsonify({
        "success": True,
        "patient": patient,
        "history": history,
        "count": len(history)
    })


@app.route("/api/patient/<patient_id>/history")
def patient_history_api(patient_id):
    if get_patient_by_id(patient_id) is None:
        return jsonify({
            "success": False,
            "message": "Patient not found."
        }), 404

    history = get_patient_history(patient_id, 100)

    return jsonify({
        "success": True,
        "patient_id": patient_id,
        "count": len(history),
        "predictions": history
    })


# ============================================================
# AI ASSISTANT
# ============================================================

def assistant_response(message, prediction=None):
    """
    Safe, prediction-aware rule-based assistant.

    It explains stored model results and general heart-health
    information. It does not provide a medical diagnosis.
    """

    question = (message or "").strip().lower()

    if not question:
        return (
            "Please enter a question. You can ask me to explain "
            "your risk score, probability, detection status, "
            "risk factors, or what to discuss with a doctor."
        )

    emergency_words = [
        "severe chest pain",
        "crushing chest pain",
        "difficulty breathing",
        "shortness of breath",
        "fainting",
        "unconscious",
        "heart attack",
        "collapse"
    ]

    if any(word in question for word in emergency_words):
        return (
            "If you are currently experiencing severe chest pain, "
            "difficulty breathing, fainting, collapse, or another "
            "serious symptom, seek urgent medical care or contact "
            "your local emergency service. This software cannot "
            "handle medical emergencies."
        )

    if prediction is None:
        return (
            "I can explain general heart-health information. "
            "For a prediction-specific explanation, open the AI "
            "Assistant from a completed assessment result."
        )

    category = prediction.get("risk_category", "Not available")
    risk_score = prediction.get("risk_score", "N/A")
    probability = prediction.get("probability", "N/A")
    detection = prediction.get("detection_status", "Not available")
    explanation = prediction.get(
        "explanation",
        "No explanation is stored."
    )

    if any(word in question for word in [
        "why", "explain", "result", "risk", "score", "probability"
    ]):
        return (
            f"The stored assessment is classified as {category} "
            f"with a risk score of {risk_score}% and a model "
            f"probability of {probability}. {explanation} "
            "The probability is a model output, not a confirmed "
            "clinical diagnosis."
        )

    if any(word in question for word in [
        "detect", "detection", "disease"
    ]):
        return (
            f"The system's current detection indication is: "
            f"{detection}. This indication is derived from the "
            "same model probability used for the assessment; it "
            "is not a separate diagnostic test."
        )

    if any(word in question for word in [
        "doctor", "physician", "hospital", "consult"
    ]):
        return (
            "You can discuss the prediction score, model probability, "
            "clinical parameters entered, symptoms, medications, "
            "medical history, and whether further clinical tests are "
            "appropriate with a qualified healthcare professional."
        )

    if any(word in question for word in [
        "diet", "food", "eat", "exercise", "lifestyle"
    ]):
        return (
            "General heart-health habits can include a balanced diet, "
            "regular physical activity appropriate for your health "
            "status, avoiding tobacco, adequate sleep, and following "
            "medical advice. Personal diet or exercise plans should "
            "be discussed with a qualified healthcare professional."
        )

    if "cholesterol" in question:
        return (
            f"The recorded cholesterol value is "
            f"{prediction.get('cholesterol', 'N/A')}. Cholesterol "
            "is one of the clinical parameters considered by the "
            "model. A clinician should interpret it together with "
            "your full medical history and laboratory reference range."
        )

    if any(word in question for word in [
        "blood pressure", "bp", "pressure"
    ]):
        return (
            f"The recorded resting blood pressure is "
            f"{prediction.get('resting_bp', 'N/A')}. Blood pressure "
            "should be interpreted using repeated measurements and "
            "clinical context rather than this software alone."
        )

    if any(word in question for word in [
        "heart rate", "pulse"
    ]):
        return (
            f"The recorded maximum heart rate is "
            f"{prediction.get('max_heart_rate', 'N/A')}. Heart-rate "
            "values depend on age, activity, medication, and clinical "
            "context."
        )

    return (
        "I can help explain this assessment. Try asking: "
        "\"Why is my risk high?\", \"Explain my probability\", "
        "\"What does the detection result mean?\", or "
        "\"What should I discuss with a doctor?\""
    )


@app.route("/api/assistant", methods=["POST"])
def ai_assistant():
    """Prediction-aware AI Assistant endpoint."""

    try:
        data = request.get_json(silent=True) or {}

        message = str(data.get("message", "")).strip()
        prediction_id = str(data.get("prediction_id", "")).strip()
        patient_id = str(data.get("patient_id", "")).strip()

        prediction = None

        if prediction_id:
            prediction = get_prediction_by_id(prediction_id)

        if prediction is None and patient_id:
            history = get_patient_history(patient_id, 1)
            if history:
                prediction = history[0]

        answer = assistant_response(message, prediction)

        return jsonify({
            "success": True,
            "answer": answer,
            "prediction_id": (
                prediction.get("prediction_id")
                if prediction else None
            ),
            "patient_id": (
                prediction.get("patient_id")
                if prediction else patient_id or None
            )
        })

    except Exception as e:
        print("AI ASSISTANT ERROR:", e)

        return jsonify({
            "success": False,
            "message": "Unable to process AI Assistant request."
        }), 500


# ============================================================
# PHASE 4 - MEDICAL REPORT ANALYZER
# ============================================================

def allowed_medical_file(filename):
    """Return True only for supported medical-report file types."""
    if not filename or "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1].lower()
    return extension in ALLOWED_MEDICAL_EXTENSIONS


def generate_medical_upload_id():
    return f"MR-{uuid.uuid4().hex[:10].upper()}"


def get_safe_medical_filename(filename, upload_id):
    safe_original = secure_filename(filename)
    extension = safe_original.rsplit(".", 1)[1].lower() if "." in safe_original else ""
    return f"{upload_id}.{extension}"


def _clean_medical_text(text):
    """Normalize OCR/PDF text without inventing or changing values."""
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = re.sub(r"[ \t]+", " ", text)
    return text


def extract_medical_report_text(file_path, extension):
    """Extract embedded PDF text first, then use Tesseract for image/scanned content."""
    extracted_text = ""
    extraction_method = []
    ocr_used = False

    if extension == "pdf":
        if pdfplumber is None:
            raise RuntimeError("pdfplumber is not installed. Run: pip install -r requirements.txt")

        pdf_text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pdf_text_parts.append(page_text)

            extracted_text = "\n".join(pdf_text_parts)

            # If the PDF is scanned/image-based, render each page and OCR it.
            if not extracted_text.strip():
                if pytesseract is None:
                    raise RuntimeError(
                        "This PDF appears to be scanned and pytesseract is not installed. "
                        "Run: pip install -r requirements.txt"
                    )

                ocr_parts = []
                for page in pdf.pages:
                    page_image = page.to_image(resolution=180).original
                    print("OCR PATH:", pytesseract.pytesseract.tesseract_cmd)
                    print("TESSERACT EXISTS:", os.path.isfile(
                    pytesseract.pytesseract.tesseract_cmd
                    ))
                    print("TESSERACT VERSION:", pytesseract.get_tesseract_version())
                    ocr_parts.append(pytesseract.image_to_string(page_image))
                extracted_text = "\n".join(ocr_parts)
                ocr_used = True

        extraction_method.append("OCR" if ocr_used else "PDF text extraction")

    else:
        if Image is None or pytesseract is None:
            raise RuntimeError(
                "Pillow/pytesseract is not installed. Run: pip install -r requirements.txt"
            )
        with Image.open(file_path) as image:
            extracted_text = pytesseract.image_to_string(image)
        extraction_method.append("OCR")
        ocr_used = True

    extracted_text = _clean_medical_text(extracted_text)
    return extracted_text, ", ".join(extraction_method), ocr_used


def _find_number(patterns, text, flags=re.IGNORECASE):
    for pattern in patterns:
        match = re.search(pattern, text, flags)
        if match:
            try:
                return float(match.group(1)), match.group(0)
            except (ValueError, IndexError):
                continue
    return None, None


def _find_blood_pressure(text):
    patterns = [
        r"(?:blood\s*pressure|bp|b\.p\.)\s*[:=\-]?\s*(\d{2,3})\s*/\s*(\d{2,3})\s*(?:mm\s*hg)?",
        r"(\d{2,3})\s*/\s*(\d{2,3})\s*mm\s*hg"
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def _status(value, low=None, high=None, lower_inclusive=True, upper_inclusive=True):
    if value is None:
        return "Not Detected"
    if low is not None and (value < low or (not lower_inclusive and value == low)):
        return "Low"
    if high is not None and (value > high or (not upper_inclusive and value == high)):
        return "High"
    return "Within Range"


def extract_medical_values(text):
    """Extract only values actually present in the uploaded report."""
    values = {}


    # Existing supported numeric values (preserved)
    hb, _ = _find_number([
        r"(?:hemoglobin|haemoglobin|hgb|hb)\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*(?:g\s*/\s*dl)?"
    ], text)
    glucose, _ = _find_number([
        r"(?:fasting\s+blood\s+glucose|fasting\s+glucose|fasting\s+blood\s+sugar|glucose)\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*(?:mg\s*/\s*dl)?"
    ], text)
    total_chol, _ = _find_number([
        r"(?:total\s+cholesterol|cholesterol)\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*(?:mg\s*/\s*dl)?"
    ], text)
    hdl, _ = _find_number([
        r"(?:hdl(?:\s+cholesterol)?)\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*(?:mg\s*/\s*dl)?"
    ], text)
    ldl, _ = _find_number([
        r"(?:ldl(?:\s+cholesterol)?)\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*(?:mg\s*/\s*dl)?"
    ], text)
    triglycerides, _ = _find_number([
        r"(?:triglycerides|triglyceride)\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*(?:mg\s*/\s*dl)?"
    ], text)
    heart_rate, _ = _find_number([
        r"(?:heart\s*rate|pulse)\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*(?:bpm|beats\s*/\s*min)?"
    ], text)
    bmi, _ = _find_number([
        r"(?:bmi|body\s+mass\s+index)\s*[:=\-]?\s*(\d+(?:\.\d+)?)"
    ], text)
    bp = _find_blood_pressure(text)

    # Added: Maximum heart rate (only explicit labels; avoids matching ordinary HR)
    max_heart_rate, _ = _find_number([
        r"(?:maximum\s+heart\s*rate|max(?:imum)?\s*hr|peak\s+heart\s*rate|max\s+heart\s*rate)\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*(?:bpm|beats\s*/\s*min)?"
    ], text)

    # Added: Oldpeak / ST depression
    oldpeak, _ = _find_number([
        # Supports labels such as "Oldpeak: 1.2", "Oldpeak (ST Depression) 1.2",
        # and "ST Depression: 1.2" while requiring an explicit label.
        r"(?:oldpeak(?:\s*\([^)]*\))?|st\s*depression|st\s*deviation)\s*[:=\-]?\s*(-?\d+(?:\.\d+)?)"
    ], text)

    # Added: categorical clinical values

    chest_pain_match = re.search(
        r"(?:chest\s*pain\s*type|chest\s*pain|cp\s*type)\s*[:=\-]?\s*"
        r"(typical\s+angina|atypical\s+angina|non[- ]?anginal(?:\s+pain)?|"
        r"asymptomatic|typical|atypical|non[- ]?anginal)",
        text, re.IGNORECASE
    )
    chest_pain = chest_pain_match.group(1).strip().title() if chest_pain_match else None

    resting_ecg_match = re.search(
        r"(?:resting\s*ecg|resting\s*electrocardiogram|ecg|ekg)\s*[:=\-]?\s*"
        r"(normal\s+sinus\s+rhythm|normal\s+or\s+st[- ]?t\s+abnormality|"
        r"st[- ]?t\s+abnormality|left\s+ventricular\s+hypertrophy|"
        r"normal|abnormal|lvh)",
        text, re.IGNORECASE
    )
    resting_ecg = resting_ecg_match.group(1).strip().title() if resting_ecg_match else None


    exercise_angina_match = re.search(
        r"(?:exercise[- ]?induced\s+angina|exercise\s+angina|"
        r"angina\s+on\s+exercise|exang)\s*[:=\-]?\s*"
        r"(yes|no|positive|negative|present|absent)",
        text, re.IGNORECASE
    )
    exercise_angina = None
    if exercise_angina_match:
        raw_angina = exercise_angina_match.group(1).strip().lower()
        if raw_angina in ("yes", "positive", "present"):
            exercise_angina = "Yes"
        elif raw_angina in ("no", "negative", "absent"):
            exercise_angina = "No"

    # Existing output structure and statuses (preserved)
    values["Hemoglobin"] = {
        "value": f"{hb:g} g/dL" if hb is not None else "Not Detected",
        "status": _status(hb, 13.0, 17.0),
        "numeric_value": hb,
        "reference_range": "13.0-17.0 g/dL"
    }
    values["Fasting Blood Glucose"] = {
        "value": f"{glucose:g} mg/dL" if glucose is not None else "Not Detected",
        "status": _status(glucose, 70.0, 100.0),
        "numeric_value": glucose,
        "reference_range": "70-100 mg/dL"
    }
    values["Total Cholesterol"] = {
        "value": f"{total_chol:g} mg/dL" if total_chol is not None else "Not Detected",
        "status": _status(total_chol, None, 200.0, upper_inclusive=False),
        "numeric_value": total_chol,
        "reference_range": "<200 mg/dL"
    }
    values["HDL Cholesterol"] = {
        "value": f"{hdl:g} mg/dL" if hdl is not None else "Not Detected",
        "status": _status(hdl, 40.0, None, lower_inclusive=False),
        "numeric_value": hdl,
        "reference_range": ">40 mg/dL"
    }
    values["LDL Cholesterol"] = {
        "value": f"{ldl:g} mg/dL" if ldl is not None else "Not Detected",
        "status": _status(ldl, None, 100.0, upper_inclusive=False),
        "numeric_value": ldl,
        "reference_range": "<100 mg/dL"
    }
    values["Triglycerides"] = {
        "value": f"{triglycerides:g} mg/dL" if triglycerides is not None else "Not Detected",
        "status": _status(triglycerides, None, 150.0, upper_inclusive=False),
        "numeric_value": triglycerides,
        "reference_range": "<150 mg/dL"
    }
    values["Blood Pressure"] = {
        "value": f"{bp[0]}/{bp[1]} mmHg" if bp else "Not Detected",
        "status": (
            "High" if bp and (bp[0] >= 120 or bp[1] >= 80)
            else "Low" if bp and (bp[0] < 90 or bp[1] < 60)
            else "Within Range" if bp else "Not Detected"
        ),
        "numeric_value": list(bp) if bp else None,
        "reference_range": "<120/80 mmHg"
    }
    values["Heart Rate"] = {
        "value": f"{heart_rate:g} bpm" if heart_rate is not None else "Not Detected",
        "status": _status(heart_rate, 60.0, 100.0),
        "numeric_value": heart_rate,
        "reference_range": "60-100 bpm"
    }
    values["BMI"] = {
        "value": f"{bmi:g} kg/m²" if bmi is not None else "Not Detected",
        "status": _status(bmi, 18.5, 24.9),
        "numeric_value": bmi,
        "reference_range": "18.5-24.9 kg/m²"
    }

    # Added outputs; non-numeric/categorical parameters are not assigned
    # medical high/low ranges by this application.
    values["Maximum Heart Rate"] = {
        "value": f"{max_heart_rate:g} bpm" if max_heart_rate is not None else "Not Detected",
        "status": "Detected" if max_heart_rate is not None else "Not Detected",
        "numeric_value": max_heart_rate,
        "reference_range": "Not configured"
    }
    values["Oldpeak"] = {
        "value": f"{oldpeak:g}" if oldpeak is not None else "Not Detected",
        "status": "Detected" if oldpeak is not None else "Not Detected",
        "numeric_value": oldpeak,
        "reference_range": "Not configured"
    }
    values["Chest Pain Type"] = {
        "value": chest_pain if chest_pain else "Not Detected",
        "status": "Detected" if chest_pain else "Not Detected",
        "numeric_value": None,
        "reference_range": "Categorical"
    }
    values["Resting ECG"] = {
        "value": resting_ecg if resting_ecg else "Not Detected",
        "status": "Detected" if resting_ecg else "Not Detected",
        "numeric_value": None,
        "reference_range": "Categorical"
    }
    values["Exercise-Induced Angina"] = {
        "value": exercise_angina if exercise_angina else "Not Detected",
        "status": "Detected" if exercise_angina else "Not Detected",
        "numeric_value": None,
        "reference_range": "Categorical"
    }

    return values

def build_medical_analysis(values):
    detected = [name for name, item in values.items() if item["status"] != "Not Detected"]
    high = [name for name, item in values.items() if item["status"] == "High"]
    low = [name for name, item in values.items() if item["status"] == "Low"]
    normal = [name for name, item in values.items() if item["status"] == "Within Range"]

    observations = []
    if high:
        observations.append("The uploaded report contains values above the configured reference range for: " + ", ".join(high) + ".")
    if low:
        observations.append("The uploaded report contains values below the configured reference range for: " + ", ".join(low) + ".")
    if not observations and normal:
        observations.append("The detected values are within the configured reference ranges.")
    if not detected:
        observations.append("No supported medical values were detected from the uploaded report.")

    explanation_parts = []
    if high:
        explanation_parts.append("Some detected values are outside the configured reference ranges and may be worth discussing with a healthcare professional.")
    if low:
        explanation_parts.append("Some detected values are below the configured reference ranges and may also require clinical context.")
    if not explanation_parts:
        explanation_parts.append("The detected values are within the configured reference ranges used by this software.")

    doctor_points = []
    for name in high + low:
        doctor_points.append(f"Discuss the {name} result and its clinical context.")
    if not doctor_points:
        doctor_points.append("Discuss the report with a qualified healthcare professional if you have symptoms, concerns, or existing medical conditions.")

    return {
        "detected_count": len(detected),
        "total_supported_values": len(values),
        "high_count": len(high),
        "low_count": len(low),
        "normal_count": len(normal),
        "observations": observations,
        "explanation": " ".join(explanation_parts),
        "doctor_discussion": doctor_points,
        "disclaimer": (
            "AI-assisted screening/explanation only. This software does not diagnose heart disease "
            "and the uploaded report does not by itself confirm or rule out any disease."
        )
    }


@app.route("/api/medical-report/upload", methods=["POST"])
def upload_medical_report():
    """Upload, extract and analyze a medical report without inventing missing values."""
    upload_id = generate_medical_upload_id()
    saved_path = None

    try:
        if "file" not in request.files:
            return jsonify({"success": False, "message": "Please select a medical report file."}), 400

        file = request.files["file"]
        if not file or not file.filename:
            return jsonify({"success": False, "message": "No file was selected."}), 400

        original_filename = file.filename
        if not allowed_medical_file(original_filename):
            return jsonify({
                "success": False,
                "message": "Unsupported file type. Use PDF, JPG, JPEG or PNG."
            }), 400

        file.stream.seek(0, os.SEEK_END)
        file_size = file.stream.tell()
        file.stream.seek(0)

        if file_size <= 0:
            return jsonify({"success": False, "message": "The uploaded file is empty."}), 400

        if file_size > MAX_MEDICAL_FILE_SIZE:
            return jsonify({"success": False, "message": "File size must not exceed 10 MB."}), 413

        server_filename = get_safe_medical_filename(original_filename, upload_id)
        saved_path = os.path.join(MEDICAL_UPLOAD_FOLDER, server_filename)
        file.save(saved_path)

        if not os.path.exists(saved_path) or os.path.getsize(saved_path) <= 0:
            raise RuntimeError("Uploaded file could not be saved correctly.")

        extension = original_filename.rsplit(".", 1)[1].lower()
        text_content, extraction_method, ocr_used = extract_medical_report_text(saved_path, extension)
        values = extract_medical_values(text_content)
        analysis = build_medical_analysis(values)
        created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        patient_id = (request.form.get("patient_id") or "").strip() or None

        save_medical_report_record({
            "upload_id": upload_id,
            "patient_id": patient_id,
            "original_filename": secure_filename(original_filename),
            "filename": server_filename,
            "file_type": extension,
            "file_size": file_size,
            "extraction_method": extraction_method,
            "ocr_used": ocr_used,
            "values": values,
            "analysis": analysis,
            "created_at": created_at,
        })

        # Keep only safe, useful metadata in the response. Raw report text is not exposed by default.
        return jsonify({
            "success": True,
            "message": "Medical report uploaded and analyzed successfully.",
            "upload_id": upload_id,
            "original_filename": secure_filename(original_filename),
            "filename": server_filename,
            "file_size": file_size,
            "file_type": extension,
            "analysis_status": "analyzed",
            "created_at": created_at,
            "extraction_method": extraction_method,
            "ocr_used": ocr_used,
            "values": values,
            "analysis": analysis
        })

    except Exception as e:
        print("OCR UPLOAD ROUTE REACHED")
        print("MEDICAL REPORT ANALYZER ERROR:", e)
        return jsonify({
            "success": False,
            "message": str(e) or "Unable to process the medical report."
        }), 500


@app.errorhandler(413)
def medical_file_too_large(error):
    return jsonify({
        "success": False,
        "message": "File size must not exceed 10 MB."
    }), 413


# ============================================================
# PHASE 5 - HISTORY & ANALYTICS HELPERS
# ============================================================

def save_medical_report_record(record):
    connection = get_db_connection()
    connection.execute(
        """
        INSERT OR REPLACE INTO medical_reports (
            upload_id, patient_id, original_filename, filename, file_type,
            file_size, extraction_method, ocr_used, values_json,
            analysis_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record["upload_id"],
            record.get("patient_id"),
            record["original_filename"],
            record["filename"],
            record["file_type"],
            record["file_size"],
            record.get("extraction_method"),
            1 if record.get("ocr_used") else 0,
            json.dumps(record.get("values", {})),
            json.dumps(record.get("analysis", {})),
            record["created_at"],
        )
    )
    connection.commit()
    connection.close()


def get_medical_report_history(limit=100, patient_id=None):
    connection = get_db_connection()
    if patient_id:
        rows = connection.execute(
            """
            SELECT * FROM medical_reports
            WHERE patient_id = ?
            ORDER BY id DESC LIMIT ?
            """,
            (patient_id, limit)
        ).fetchall()
    else:
        rows = connection.execute(
            "SELECT * FROM medical_reports ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
    connection.close()

    results = []
    for row in rows:
        item = dict(row)
        try:
            item["values"] = json.loads(item.pop("values_json") or "{}")
        except (TypeError, ValueError):
            item["values"] = {}
        try:
            item["analysis"] = json.loads(item.pop("analysis_json") or "{}")
        except (TypeError, ValueError):
            item["analysis"] = {}
        item["ocr_used"] = bool(item.get("ocr_used"))
        results.append(item)
    return results


def get_history_analytics():
    connection = get_db_connection()
    total_predictions = connection.execute("SELECT COUNT(*) AS count FROM predictions").fetchone()["count"]
    avg_row = connection.execute("SELECT AVG(risk_score) AS avg_score FROM predictions").fetchone()
    total_patients = connection.execute("SELECT COUNT(*) AS count FROM patients").fetchone()["count"]
    total_reports = connection.execute("SELECT COUNT(*) AS count FROM medical_reports").fetchone()["count"]
    report_high = connection.execute(
        "SELECT COUNT(*) AS count FROM medical_reports WHERE analysis_json LIKE '%\"high_count\":% '"
    ).fetchone()["count"]

    trend_rows = connection.execute(
        """
        SELECT DATE(created_at) AS date, COUNT(*) AS count, ROUND(AVG(risk_score), 1) AS avg_risk
        FROM predictions
        GROUP BY DATE(created_at)
        ORDER BY DATE(created_at) ASC
        LIMIT 30
        """
    ).fetchall()
    connection.close()

    risk_counts = get_dashboard_statistics()
    return {
        "total_predictions": total_predictions,
        "total_patients": total_patients,
        "total_medical_reports": total_reports,
        "average_risk_score": round(float(avg_row["avg_score"]), 1) if avg_row["avg_score"] is not None else 0,
        "risk_counts": risk_counts,
        "prediction_trend": [dict(row) for row in trend_rows],
    }


@app.route("/api/history-data")
def history_data_api():
    return jsonify({"success": True, "count": len(get_recent_predictions(100)), "predictions": get_recent_predictions(100)})


@app.route("/api/history-analytics")
def history_analytics_api():
    return jsonify({"success": True, **get_history_analytics()})


@app.route("/api/medical-report/history")
def medical_report_history_api():
    patient_id = request.args.get("patient_id", "").strip() or None
    reports = get_medical_report_history(100, patient_id)
    return jsonify({"success": True, "count": len(reports), "reports": reports})


# ============================================================
# DASHBOARD API
# ============================================================

@app.route("/api/dashboard")
def dashboard_api():
    """
    Dashboard statistics API.
    """

    stats = get_dashboard_statistics()

    recent = get_recent_predictions(10)

    connection = get_db_connection()
    patient_count = connection.execute(
        "SELECT COUNT(*) AS count FROM patients"
    ).fetchone()["count"]
    connection.close()

    return jsonify({
        "success": True,
        "statistics": stats,
        "recent_predictions": recent,
        "patient_statistics": {
            "total_patients": patient_count
        }
    })


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/api/health")
def health_check():
    """
    Application health endpoint.

    Useful for deployment and debugging.
    """

    return jsonify({
        "status": "healthy",
        "application": (
            "AI-Driven Heart Disease Prediction, "
            "Detection, and Classification System"
        ),
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "database": os.path.exists(DATABASE_PATH),
        "patient_management": True,
        "medical_report_analyzer": True,
        "medical_report_upload_limit_mb": 10,
        "medical_report_history": True,
        "history_analytics": True,
        "ai_assistant": True,
        "timestamp": datetime.datetime.now().isoformat()
    })



# ============================================================
# PHASE 7 — AUTHENTICATION, ROLES & DOCTOR VERIFICATION
# ============================================================

from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash

# NOTE: Phase 7 is additive. Existing prediction/OCR/history/XAI/PDF code remains unchanged.
app.secret_key = os.environ.get("HEART_AI_SECRET_KEY", "change-this-secret-key-before-deployment")

AUTH_UPLOAD_FOLDER = os.path.join(BASE_DIR, "doctor_documents")
os.makedirs(AUTH_UPLOAD_FOLDER, exist_ok=True)
AUTH_ALLOWED_DOCUMENTS = {"pdf", "jpg", "jpeg", "png"}

# Phase 7 — Super Admin / Admin OTP configuration
SUPER_ADMIN_EMAIL = os.environ.get(
    "HEART_AI_SUPERADMIN_EMAIL",
    "vermaakul9642@gmail.com"
).strip().lower()
OTP_EXPIRY_MINUTES = int(os.environ.get("HEART_AI_OTP_EXPIRY_MINUTES", "10"))
OTP_MAX_ATTEMPTS = int(os.environ.get("HEART_AI_OTP_MAX_ATTEMPTS", "5"))
SMTP_HOST = os.environ.get("HEART_AI_SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("HEART_AI_SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("HEART_AI_SMTP_USERNAME", "").strip()
SMTP_PASSWORD = os.environ.get("HEART_AI_SMTP_PASSWORD", "")
SMTP_FROM = os.environ.get("HEART_AI_SMTP_FROM", SMTP_USERNAME).strip()


def initialize_auth_database():
    connection = get_db_connection()
    connection.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('PATIENT','DOCTOR','ADMIN')),
            patient_id TEXT,
            doctor_id TEXT UNIQUE,
            full_name TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            is_active INTEGER DEFAULT 1
        )
    """)
    # Existing users table is preserved; add admin_level only if it is missing.
    user_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(users)").fetchall()
    }
    if "admin_level" not in user_columns:
        connection.execute("ALTER TABLE users ADD COLUMN admin_level TEXT DEFAULT 'ADMIN'")

    connection.execute("""
        CREATE TABLE IF NOT EXISTS admin_otps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            otp_hash TEXT NOT NULL,
            purpose TEXT NOT NULL,
            invited_by INTEGER,
            full_name TEXT,
            expires_at TEXT NOT NULL,
            attempts INTEGER DEFAULT 0,
            used INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)
    connection.execute("CREATE INDEX IF NOT EXISTS idx_admin_otps_email ON admin_otps(email)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_admin_otps_expires ON admin_otps(expires_at)")

    connection.execute("""
        CREATE TABLE IF NOT EXISTS doctors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER UNIQUE NOT NULL,
            doctor_id TEXT UNIQUE NOT NULL,
            professional_registration_no TEXT UNIQUE NOT NULL,
            issuing_authority TEXT NOT NULL,
            qualification TEXT NOT NULL,
            specialization TEXT,
            institution TEXT,
            document_filename TEXT,
            verification_status TEXT NOT NULL DEFAULT 'PENDING',
            verified_by INTEGER,
            verified_at TEXT,
            rejection_reason TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            admin_user_id INTEGER,
            action TEXT NOT NULL,
            target_type TEXT,
            target_id TEXT,
            details TEXT,
            created_at TEXT NOT NULL
        )
    """)
    # Phase 7 — Doctor Patient Activity History
    # Every successful doctor patient lookup is recorded.
    connection.execute("""
        CREATE TABLE IF NOT EXISTS doctor_patient_activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_user_id INTEGER NOT NULL,
            doctor_id TEXT NOT NULL,
            patient_id TEXT NOT NULL,
            action TEXT NOT NULL,
            prediction_id TEXT,
            medical_report_id TEXT,
            details TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(doctor_user_id) REFERENCES users(id)
        )
    """)
    connection.execute("CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_doctors_status ON doctors(verification_status)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_logs(created_at)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_doctor_activity_doctor ON doctor_patient_activity(doctor_user_id)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_doctor_activity_patient ON doctor_patient_activity(patient_id)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_doctor_activity_created ON doctor_patient_activity(created_at)")

    # The initial Super Admin is identified by the configured authorized email.
    # No public Admin signup and no hard-coded admin password are created.
    existing_super = connection.execute(
        "SELECT id FROM users WHERE email=? AND role='ADMIN'",
        (SUPER_ADMIN_EMAIL,)
    ).fetchone()
    if existing_super:
        connection.execute(
            "UPDATE users SET admin_level='SUPER_ADMIN' WHERE id=?",
            (existing_super["id"],)
        )
    connection.commit()
    print("Configured Super Admin email:", SUPER_ADMIN_EMAIL)
    connection.close()


initialize_auth_database()


def current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    connection = get_db_connection()
    row = connection.execute("SELECT * FROM users WHERE id=? AND is_active=1", (user_id,)).fetchone()
    connection.close()
    return dict(row) if row else None


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if current_user() is None:
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def role_required(*roles):
    allowed = {role.upper() for role in roles}
    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            user = current_user()
            if user is None:
                return redirect(url_for("login"))
            if user["role"] not in allowed:
                return ("<h2>Access denied</h2><p>You are not authorized to access this page.</p>", 403)
            return view(*args, **kwargs)
        return wrapped
    return decorator


def generate_doctor_id():
    return f"DOC-{uuid.uuid4().hex[:8].upper()}"


def log_admin_action(admin_user_id, action, target_type=None, target_id=None, details=None):
    connection = get_db_connection()
    connection.execute("""
        INSERT INTO audit_logs (admin_user_id, action, target_type, target_id, details, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (admin_user_id, action, target_type, target_id, details or "", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    connection.commit()
    connection.close()


def hash_admin_otp(otp):
    return hashlib.sha256(otp.encode("utf-8")).hexdigest()


def send_admin_otp(email, otp, purpose):
    """Send an administrator OTP using configured SMTP credentials."""
    if not SMTP_USERNAME or not SMTP_PASSWORD or not SMTP_FROM:
        raise RuntimeError(
            "SMTP is not configured. Set HEART_AI_SMTP_USERNAME, "
            "HEART_AI_SMTP_PASSWORD and HEART_AI_SMTP_FROM before using Admin OTP."
        )

    subject = "Heart AI — Administrator Email Verification"
    purpose_text = (
        "initial Super Admin activation"
        if purpose == "INITIAL_SUPER_ADMIN"
        else "new Administrator invitation"
    )
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = SMTP_FROM
    message["To"] = email
    message.set_content(
        f"Heart AI Administrator Verification\n\n"
        f"Purpose: {purpose_text}\n\n"
        f"Your one-time verification code is: {otp}\n\n"
        f"This code expires in {OTP_EXPIRY_MINUTES} minutes.\n"
        f"If you did not request administrator access, ignore this email."
    )

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as smtp:
        smtp.starttls()
        smtp.login(SMTP_USERNAME, SMTP_PASSWORD)
        smtp.send_message(message)


def create_admin_otp(email, purpose, invited_by=None, full_name=None):
    otp = f"{secrets.randbelow(1000000):06d}"
    now = datetime.datetime.now()
    expires = now + datetime.timedelta(minutes=OTP_EXPIRY_MINUTES)
    connection = get_db_connection()
    connection.execute(
        "UPDATE admin_otps SET used=1 WHERE email=? AND used=0",
        (email,)
    )
    connection.execute("""
        INSERT INTO admin_otps
        (email, otp_hash, purpose, invited_by, full_name, expires_at, attempts, used, created_at)
        VALUES (?, ?, ?, ?, ?, ?, 0, 0, ?)
    """, (
        email, hash_admin_otp(otp), purpose, invited_by, full_name,
        expires.strftime("%Y-%m-%d %H:%M:%S"),
        now.strftime("%Y-%m-%d %H:%M:%S")
    ))
    connection.commit()
    connection.close()
    send_admin_otp(email, otp, purpose)


def verify_admin_otp(email, otp):
    connection = get_db_connection()
    row = connection.execute("""
        SELECT * FROM admin_otps
        WHERE email=? AND used=0
        ORDER BY id DESC LIMIT 1
    """, (email,)).fetchone()
    if row is None:
        connection.close()
        return None, "No active OTP found. Please request a new code."

    if row["attempts"] >= OTP_MAX_ATTEMPTS:
        connection.close()
        return None, "Too many incorrect OTP attempts. Please request a new code."

    expires = datetime.datetime.strptime(row["expires_at"], "%Y-%m-%d %H:%M:%S")
    if datetime.datetime.now() > expires:
        connection.execute("UPDATE admin_otps SET used=1 WHERE id=?", (row["id"],))
        connection.commit(); connection.close()
        return None, "OTP has expired. Please request a new code."

    if not secrets.compare_digest(row["otp_hash"], hash_admin_otp(otp)):
        connection.execute("UPDATE admin_otps SET attempts=attempts+1 WHERE id=?", (row["id"],))
        connection.commit(); connection.close()
        return None, "Incorrect OTP."

    connection.execute("UPDATE admin_otps SET used=1 WHERE id=?", (row["id"],))
    connection.commit(); connection.close()
    return dict(row), None


def admin_dashboard_redirect(user):
    if user.get("role") == "ADMIN":
        return redirect(url_for("admin_dashboard"))
    return redirect(url_for("dashboard"))


# Authentication endpoints are additive; existing endpoints remain in the file.
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        # First-time Super Admin: the configured email alone can start OTP activation.
        connection = get_db_connection()
        user = connection.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        connection.close()

        if email == SUPER_ADMIN_EMAIL and user is None:
            try:
                create_admin_otp(email, "INITIAL_SUPER_ADMIN", full_name="Super Administrator")
            except Exception as exc:
                return render_template(
                    "login.html",
                    error=f"Unable to send Super Admin OTP. Configure SMTP first. Details: {exc}",
                    email=email
                )
            return redirect(url_for("admin_verify", email=email, purpose="INITIAL_SUPER_ADMIN"))

        if user is None or not user["is_active"] or not check_password_hash(user["password_hash"], password):
            return render_template("login.html", error="Invalid email or password.", email=email)

        if user["role"] == "DOCTOR":
            connection = get_db_connection()
            doctor = connection.execute("SELECT verification_status FROM doctors WHERE user_id=?", (user["id"],)).fetchone()
            connection.close()
            if doctor is None or doctor["verification_status"] != "VERIFIED":
                return render_template("login.html", error="Doctor account is not verified by Admin yet.", email=email)

        session.clear()
        session["user_id"] = user["id"]
        session["role"] = user["role"]
        session["admin_level"] = user["admin_level"] if "admin_level" in user.keys() else None
        session["full_name"] = user["full_name"] or ""
        session["patient_id"] = user["patient_id"]
        session["doctor_id"] = user["doctor_id"]

        if user["role"] == "ADMIN":
            return redirect(url_for("admin_dashboard"))
        if user["role"] == "DOCTOR":
            return redirect(url_for("doctor_dashboard"))
        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/admin/verify", methods=["GET", "POST"])
def admin_verify():
    email = request.args.get("email", "").strip().lower() or request.form.get("email", "").strip().lower()
    purpose = request.args.get("purpose", "ADMIN_INVITE")
    if not email:
        return redirect(url_for("login"))

    if request.method == "POST":
        otp = request.form.get("otp", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        if not re.fullmatch(r"\d{6}", otp):
            return render_template("admin_verify.html", email=email, error="Enter the 6-digit OTP.")
        if len(password) < 8:
            return render_template("admin_verify.html", email=email, error="Password must contain at least 8 characters.")
        if password != confirm_password:
            return render_template("admin_verify.html", email=email, error="Passwords do not match.")

        otp_row, error = verify_admin_otp(email, otp)
        if error:
            return render_template("admin_verify.html", email=email, error=error)

        connection = get_db_connection()
        try:
            existing = connection.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if existing:
                # Never allow an invited account to overwrite a different role.
                if existing["role"] != "ADMIN":
                    connection.rollback(); connection.close()
                    return render_template("admin_verify.html", email=email, error="This email is already registered as a non-admin account.")
                connection.execute("""
                    UPDATE users
                    SET password_hash=?, role='ADMIN', admin_level=?, full_name=?, is_active=1, updated_at=?
                    WHERE id=?
                """, (generate_password_hash(password), "SUPER_ADMIN" if purpose == "INITIAL_SUPER_ADMIN" else "ADMIN", otp_row.get("full_name") or existing["full_name"] or "Administrator", now, existing["id"]))
                user_id = existing["id"]
            else:
                level = "SUPER_ADMIN" if purpose == "INITIAL_SUPER_ADMIN" else "ADMIN"
                cursor = connection.execute("""
                    INSERT INTO users (email, password_hash, role, admin_level, full_name, created_at, updated_at, is_active)
                    VALUES (?, ?, 'ADMIN', ?, ?, ?, ?, 1)
                """, (email, generate_password_hash(password), level, otp_row.get("full_name") or "Administrator", now, now))
                user_id = cursor.lastrowid
            connection.commit()
        except sqlite3.IntegrityError:
            connection.rollback(); connection.close()
            return render_template("admin_verify.html", email=email, error="This email is already registered.")
        connection.close()

        session.clear()
        session["user_id"] = user_id
        session["role"] = "ADMIN"
        session["admin_level"] = "SUPER_ADMIN" if purpose == "INITIAL_SUPER_ADMIN" else "ADMIN"
        session["full_name"] = otp_row.get("full_name") or "Administrator"
        session["patient_id"] = None
        session["doctor_id"] = None
        return redirect(url_for("admin_dashboard"))

    return render_template("admin_verify.html", email=email, purpose=purpose)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register_patient():
    if request.method == "POST":
        name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not name or not email or len(password) < 8:
            return render_template("register.html", error="Name, email and a password of at least 8 characters are required.")
        connection = get_db_connection()
        try:
            patient_id = get_or_create_patient("", name)
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            connection.execute("""
                INSERT INTO users (email, password_hash, role, patient_id, full_name, created_at, updated_at, is_active)
                VALUES (?, ?, 'PATIENT', ?, ?, ?, ?, 1)
            """, (email, generate_password_hash(password), patient_id, name, now, now))
            connection.commit()
        except sqlite3.IntegrityError:
            connection.rollback()
            connection.close()
            return render_template("register.html", error="An account with this email already exists.")
        connection.close()
        return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/doctor/register", methods=["GET", "POST"])
def doctor_register():
    if request.method == "POST":
        name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        reg_no = request.form.get("professional_registration_no", "").strip()
        authority = request.form.get("issuing_authority", "").strip()
        qualification = request.form.get("qualification", "").strip()
        specialization = request.form.get("specialization", "").strip()
        institution = request.form.get("institution", "").strip()
        document = request.files.get("document")
        if not all([name, email, password, reg_no, authority, qualification]):
            return render_template("doctor_register.html", error="Please fill all required fields.")
        if len(password) < 8:
            return render_template("doctor_register.html", error="Password must contain at least 8 characters.")
        if document and document.filename:
            ext = document.filename.rsplit(".", 1)[-1].lower() if "." in document.filename else ""
            if ext not in AUTH_ALLOWED_DOCUMENTS:
                return render_template("doctor_register.html", error="Document must be PDF, JPG, JPEG or PNG.")
            safe_name = secure_filename(document.filename)
            document_filename = f"{uuid.uuid4().hex}_{safe_name}"
            document.save(os.path.join(AUTH_UPLOAD_FOLDER, document_filename))
        else:
            document_filename = None
        connection = get_db_connection()
        try:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            doctor_id = generate_doctor_id()
            cursor = connection.execute("""
                INSERT INTO users (email, password_hash, role, doctor_id, full_name, created_at, updated_at, is_active)
                VALUES (?, ?, 'DOCTOR', ?, ?, ?, ?, 1)
            """, (email, generate_password_hash(password), doctor_id, name, now, now))
            user_id = cursor.lastrowid
            connection.execute("""
                INSERT INTO doctors (user_id, doctor_id, professional_registration_no, issuing_authority, qualification, specialization, institution, document_filename, verification_status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, ?)
            """, (user_id, doctor_id, reg_no, authority, qualification, specialization, institution, document_filename, now, now))
            connection.commit()
        except sqlite3.IntegrityError:
            connection.rollback()
            connection.close()
            return render_template("doctor_register.html", error="Email or professional registration number is already registered.")
        connection.close()
        return render_template("doctor_register.html", success="Registration submitted. Your account will become active after Admin verification.")
    return render_template("doctor_register.html")


@app.route("/admin")
@role_required("ADMIN")
def admin_dashboard():
    connection = get_db_connection()
    pending = [dict(r) for r in connection.execute("""
        SELECT d.*, u.email, u.full_name FROM doctors d JOIN users u ON u.id=d.user_id
        WHERE d.verification_status='PENDING' ORDER BY d.id DESC
    """).fetchall()]
    doctors = [dict(r) for r in connection.execute("""
        SELECT d.*, u.email, u.full_name FROM doctors d JOIN users u ON u.id=d.user_id
        ORDER BY d.id DESC LIMIT 100
    """).fetchall()]
    users = [dict(r) for r in connection.execute("SELECT id,email,role,full_name,patient_id,doctor_id,is_active,created_at FROM users ORDER BY id DESC LIMIT 100").fetchall()]
    logs = [dict(r) for r in connection.execute("SELECT * FROM audit_logs ORDER BY id DESC LIMIT 100").fetchall()]
    connection.close()
    return render_template("admin_dashboard.html", pending_doctors=pending, doctors=doctors, users=users, logs=logs)


@app.route("/admin/doctor/<doctor_id>/approve", methods=["POST"])
@role_required("ADMIN")
def approve_doctor(doctor_id):
    user = current_user()
    connection = get_db_connection()
    row = connection.execute("SELECT user_id FROM doctors WHERE doctor_id=?", (doctor_id,)).fetchone()
    if row is None:
        connection.close()
        return ("Doctor not found", 404)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    connection.execute("UPDATE doctors SET verification_status='VERIFIED', verified_by=?, verified_at=?, updated_at=?, rejection_reason=NULL WHERE doctor_id=?", (user["id"], now, now, doctor_id))
    connection.commit(); connection.close()
    log_admin_action(user["id"], "APPROVE_DOCTOR", "DOCTOR", doctor_id)
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/doctor/<doctor_id>/reject", methods=["POST"])
@role_required("ADMIN")
def reject_doctor(doctor_id):
    user = current_user()
    reason = request.form.get("reason", "Rejected by Admin").strip() or "Rejected by Admin"
    connection = get_db_connection()
    row = connection.execute("SELECT user_id FROM doctors WHERE doctor_id=?", (doctor_id,)).fetchone()
    if row is None:
        connection.close()
        return ("Doctor not found", 404)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    connection.execute("UPDATE doctors SET verification_status='REJECTED', verified_by=?, verified_at=?, updated_at=?, rejection_reason=? WHERE doctor_id=?", (user["id"], now, now, reason, doctor_id))
    connection.commit(); connection.close()
    log_admin_action(user["id"], "REJECT_DOCTOR", "DOCTOR", doctor_id, reason)
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/doctor/<doctor_id>/status", methods=["POST"])
@role_required("ADMIN")
def toggle_doctor_status(doctor_id):
    user = current_user()
    new_status = request.form.get("status", "").upper()
    if new_status not in {"ACTIVE", "INACTIVE"}:
        return ("Invalid status", 400)
    connection = get_db_connection()
    row = connection.execute("SELECT user_id FROM doctors WHERE doctor_id=?", (doctor_id,)).fetchone()
    if row is None:
        connection.close(); return ("Doctor not found", 404)
    connection.execute("UPDATE users SET is_active=? WHERE id=?", (1 if new_status == "ACTIVE" else 0, row["user_id"]))
    connection.commit(); connection.close()
    log_admin_action(user["id"], "SET_DOCTOR_" + new_status, "DOCTOR", doctor_id)
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/admins/create", methods=["POST"])
@role_required("ADMIN")
def create_admin():
    # Existing Admins can invite another administrator. The invitee must verify email by OTP and set a password.
    user = current_user()
    email = request.form.get("email", "").strip().lower()
    name = request.form.get("full_name", "").strip() or "Administrator"
    if not email or "@" not in email:
        return ("Valid administrator email is required.", 400)
    if email == SUPER_ADMIN_EMAIL:
        return ("This email is reserved for the configured Super Admin.", 400)

    connection = get_db_connection()
    existing = connection.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
    connection.close()
    if existing:
        return ("This email is already registered.", 409)

    try:
        create_admin_otp(email, "ADMIN_INVITE", invited_by=user["id"], full_name=name)
    except Exception as exc:
        return (f"Unable to send administrator OTP. Configure SMTP first. Details: {exc}", 500)

    log_admin_action(user["id"], "INVITE_ADMIN_OTP", "USER", email, "Administrator invitation OTP sent.")
    return redirect(url_for("admin_verify", email=email, purpose="ADMIN_INVITE"))


@app.route("/doctor")
@role_required("DOCTOR")
def doctor_dashboard():
    user = current_user()
    connection = get_db_connection()
    doctor = connection.execute("SELECT * FROM doctors WHERE user_id=?", (user["id"],)).fetchone()
    connection.close()
    activities = get_doctor_activity(user["id"], 20)
    return render_template(
        "doctor_dashboard.html",
        doctor=dict(doctor) if doctor else None,
        activities=activities
    )


# ============================================================
# PHASE 7 — PATIENT ACCESS & DOCTOR ACTIVITY
# ============================================================

def log_doctor_patient_activity(doctor_user, patient_id, action,
                                prediction_id=None, medical_report_id=None,
                                details=None):
    """Record every patient access made by a verified doctor."""
    if not doctor_user or doctor_user.get("role") != "DOCTOR":
        return

    connection = get_db_connection()
    connection.execute("""
        INSERT INTO doctor_patient_activity
        (
            doctor_user_id,
            doctor_id,
            patient_id,
            action,
            prediction_id,
            medical_report_id,
            details,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        doctor_user["id"],
        doctor_user.get("doctor_id") or "",
        patient_id,
        action,
        prediction_id,
        medical_report_id,
        details or "",
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    connection.commit()
    connection.close()


def get_doctor_activity(doctor_user_id, limit=100):
    connection = get_db_connection()
    rows = connection.execute("""
        SELECT a.*, p.patient_name
        FROM doctor_patient_activity a
        LEFT JOIN patients p ON p.patient_id = a.patient_id
        WHERE a.doctor_user_id = ?
        ORDER BY a.id DESC
        LIMIT ?
    """, (doctor_user_id, limit)).fetchall()
    connection.close()
    return [dict(row) for row in rows]


def get_patient_medical_reports(patient_id, limit=100):
    connection = get_db_connection()
    rows = connection.execute("""
        SELECT id, upload_id, patient_id, original_filename, file_type,
               file_size, extraction_method, ocr_used, values_json,
               analysis_json, created_at
        FROM medical_reports
        WHERE patient_id = ?
        ORDER BY id DESC
        LIMIT ?
    """, (patient_id, limit)).fetchall()
    connection.close()
    return [dict(row) for row in rows]


@app.route("/doctor/patient", methods=["GET", "POST"])
@role_required("DOCTOR")
def doctor_patient_lookup():
    """Doctor checks a Patient ID; every successful check is logged."""
    user = current_user()
    patient = None
    predictions = []
    reports = []
    searched_patient_id = ""

    if request.method == "POST":
        searched_patient_id = request.form.get("patient_id", "").strip().upper()

        if not searched_patient_id:
            return render_template(
                "doctor_patient.html",
                error="Please enter a Patient ID.",
                patient=None,
                predictions=[],
                reports=[],
                searched_patient_id=""
            )

        patient = get_patient_by_id(searched_patient_id)

        if patient is None:
            return render_template(
                "doctor_patient.html",
                error="Patient ID was not found.",
                patient=None,
                predictions=[],
                reports=[],
                searched_patient_id=searched_patient_id
            )

        predictions = get_patient_history(searched_patient_id, 100)
        reports = get_patient_medical_reports(searched_patient_id, 100)

        log_doctor_patient_activity(
            user,
            searched_patient_id,
            "VIEW_PATIENT",
            details="Doctor opened patient profile, prediction history and medical-report history."
        )

    return render_template(
        "doctor_patient.html",
        patient=patient,
        predictions=predictions,
        reports=reports,
        searched_patient_id=searched_patient_id
    )


@app.route("/doctor/history")
@role_required("DOCTOR")
def doctor_history():
    """Doctor can see only their own patient-check/activity history."""
    user = current_user()
    return render_template(
        "doctor_history.html",
        activities=get_doctor_activity(user["id"], 200)
    )


@app.route("/doctor/api/patient/<patient_id>")
@role_required("DOCTOR")
def doctor_patient_api(patient_id):
    """Doctor-only patient lookup API; successful access is logged."""
    user = current_user()
    patient_id = (patient_id or "").strip().upper()
    patient = get_patient_by_id(patient_id)

    if patient is None:
        return jsonify({"success": False, "message": "Patient not found."}), 404

    predictions = get_patient_history(patient_id, 100)
    reports = get_patient_medical_reports(patient_id, 100)

    log_doctor_patient_activity(
        user,
        patient_id,
        "VIEW_PATIENT_API",
        details="Doctor accessed patient data through the API."
    )

    return jsonify({
        "success": True,
        "patient": patient,
        "predictions": predictions,
        "medical_reports": reports,
        "doctor_id": user.get("doctor_id")
    })


@app.route("/admin/patients")
@role_required("ADMIN")
def admin_patients():
    """Admin can view all registered patients."""
    connection = get_db_connection()
    patients = [
        dict(row)
        for row in connection.execute("""
            SELECT p.*, u.email, u.is_active
            FROM patients p
            LEFT JOIN users u
              ON u.patient_id = p.patient_id AND u.role = 'PATIENT'
            ORDER BY p.updated_at DESC
        """).fetchall()
    ]
    connection.close()
    return render_template("admin_patients.html", patients=patients)


@app.route("/admin/patient/<patient_id>")
@role_required("ADMIN")
def admin_patient_details(patient_id):
    """Admin has system-wide access to one patient's complete record."""
    patient_id = (patient_id or "").strip().upper()
    patient = get_patient_by_id(patient_id)

    if patient is None:
        return ("<h2>Patient not found</h2><p>The requested Patient ID does not exist.</p>", 404)

    predictions = get_patient_history(patient_id, 200)
    reports = get_patient_medical_reports(patient_id, 200)

    connection = get_db_connection()
    doctor_activity = [
        dict(row)
        for row in connection.execute("""
            SELECT a.*, u.full_name AS doctor_name, u.email AS doctor_email
            FROM doctor_patient_activity a
            LEFT JOIN users u ON u.id = a.doctor_user_id
            WHERE a.patient_id = ?
            ORDER BY a.id DESC
            LIMIT 200
        """, (patient_id,)).fetchall()
    ]
    connection.close()

    return render_template(
        "admin_patient.html",
        patient=patient,
        predictions=predictions,
        reports=reports,
        doctor_activity=doctor_activity
    )


@app.before_request
def phase7_access_control():
    """Protect application pages while leaving login/registration/health public."""
    public_endpoints = {"login", "logout", "register_patient", "doctor_register", "health_check", "static"}
    endpoint = request.endpoint or ""
    if endpoint in public_endpoints or endpoint.startswith("static"):
        return None
    user = current_user()
    if user is None:
        return redirect(url_for("login"))
    # Patient/doctor data access: patient can access only their own patient context.
    if user["role"] == "PATIENT" and request.path.startswith("/api/patient/"):
        requested = request.view_args.get("patient_id") if request.view_args else request.args.get("patient_id")
        if requested and requested != user["patient_id"]:
            return jsonify({"success": False, "message": "Access denied."}), 403
    if user["role"] == "PATIENT" and request.path == "/api/medical-report/history":
        requested = request.args.get("patient_id")
        if requested and requested != user["patient_id"]:
            return jsonify({"success": False, "message": "Access denied."}), 403
    return None


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def page_not_found(error):
    """
    Custom 404 response.
    """

    return (
        """
        <html>
            <head>
                <title>Page Not Found</title>
            </head>
            <body>
                <h1>404 - Page Not Found</h1>
                <p>The requested page does not exist.</p>
                <a href="/prediction">Return to Home</a>
            </body>
        </html>
        """,
        404
    )


@app.errorhandler(500)
def internal_server_error(error):
    """
    Custom 500 response.
    """

    return (
        """
        <html>
            <head>
                <title>Server Error</title>
            </head>
            <body>
                <h1>500 - Internal Server Error</h1>
                <p>Something went wrong on the server.</p>
                <a href="/prediction">Return to Home</a>
            </body>
        </html>
        """,
        500
    )


# ============================================================
# APPLICATION START
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )