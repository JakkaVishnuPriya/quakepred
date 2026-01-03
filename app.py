from flask import Flask, request, render_template, redirect, url_for, flash, session
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import uuid
import os

# ======================================================
#                     APP SETUP
# ======================================================
app = Flask(__name__)
app.secret_key = "super_secret_key_for_session"

# ======================================================
#                 PATHS (RENDER SAFE)
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "random_forest_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, "feature_names.pkl")

# ======================================================
#                LOAD MODEL FILES
# ======================================================
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    print("✅ Model, scaler, and feature names loaded successfully.")
except Exception as e:
    print("❌ Model loading failed:", e)
    raise RuntimeError("Model or scaler files missing")

# ======================================================
#                 ALERT MAPPINGS
# ======================================================
alert_to_class = {
    "green": 0,
    "yellow": 3,
    "orange": 1,
    "red": 2
}
class_to_alert = {v: k for k, v in alert_to_class.items()}

ALERT_COLORS = {
    "green": "#10b981",
    "yellow": "#f59e0b",
    "orange": "#f97316",
    "red": "#ef4444"
}

# ======================================================
#                      ROUTES
# ======================================================

# ---------- LANDING ----------
@app.route("/")
def index():
    return render_template("index.html")

# ---------- LOGIN ----------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        # ✅ DEMO CREDENTIALS
        if username == "demo_user" and password == "demo123":
            session["logged_in"] = True
            session["username"] = username
            flash("Logged in successfully!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials. Use demo_user / demo123", "danger")

    return render_template("login.html")

# ---------- SIGNUP ----------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        flash("Signup successful! Please login.", "success")
        return redirect(url_for("login"))
    return render_template("signup.html")

# ---------- DASHBOARD (PROTECTED) ----------
@app.route("/dashboard")
def dashboard():
    if not session.get("logged_in"):
        flash("Please login to continue.", "warning")
        return redirect(url_for("login"))
    return render_template("dashboard.html", username=session.get("username"))

# ---------- FREE PREDICTION ----------
@app.route("/free_prediction")
def free_prediction():
    return render_template("free_prediction.html")

# ---------- ADVANCED PREDICTION ----------
@app.route("/predict")
def predict():
    if not session.get("logged_in"):
        flash("Login required for advanced prediction.", "warning")
        return redirect(url_for("login"))
    return render_template("predict.html")

# ======================================================
#                MAKE PREDICTION
# ======================================================
@app.route("/make_prediction", methods=["POST"])
def make_prediction():
    try:
        magnitude = float(request.form["magnitude"])
        depth = float(request.form["depth"])
        cdi = float(request.form["cdi"])
        mmi = float(request.form["mmi"])
        sig = float(request.form["sig"])

        input_data = pd.DataFrame([{
            "magnitude": magnitude,
            "depth": depth,
            "cdi": cdi,
            "mmi": mmi,
            "sig": sig
        }])

        input_data = input_data[feature_names]
        scaled_input = scaler.transform(input_data)

        predicted_class = model.predict(scaled_input)[0]
        probabilities = model.predict_proba(scaled_input)[0]

        predicted_alert = class_to_alert[predicted_class]

        # -------- Probability Plot --------
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = [class_to_alert[i].capitalize() for i in range(len(probabilities))]
        colors = [ALERT_COLORS[class_to_alert[i]] for i in range(len(probabilities))]

        ax.bar(labels, probabilities, color=colors)
        ax.set_ylim(0, 1)

        for i, p in enumerate(probabilities):
            ax.text(i, p + 0.02, f"{p*100:.1f}%", ha="center")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        probability_plot = base64.b64encode(buf.getvalue()).decode()

        return render_template(
            "result.html",
            prediction_result={
                "alert": predicted_alert.upper(),
                "color": ALERT_COLORS[predicted_alert],
                "probabilities": {
                    labels[i]: f"{probabilities[i]*100:.2f}%"
                    for i in range(len(probabilities))
                }
            },
            probability_plot=probability_plot,
            current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            prediction_id=str(uuid.uuid4())[:8],
            predicted_alert_class_name=predicted_alert
        )

    except Exception as e:
        return render_template("result.html", error=str(e))

# ---------- LOGOUT ----------
@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))

# ======================================================
#                 LOCAL RUN ONLY
# ======================================================
if __name__ == "__main__":
    app.run(debug=True)
