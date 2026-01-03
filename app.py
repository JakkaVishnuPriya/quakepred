from flask import Flask, render_template, request, redirect, url_for, session, flash
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import os
from datetime import datetime
import uuid

# -------------------- APP INIT --------------------
app = Flask(__name__)
app.secret_key = "quakepred_secret_key"

# -------------------- PATH SETUP (RENDER SAFE) --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, 'random_forest_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, 'feature_names.pkl')

# -------------------- LOAD MODEL FILES --------------------
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    print("✅ Model, scaler, and feature names loaded successfully.")
except Exception as e:
    print("❌ Error loading model files:", e)
    raise RuntimeError("Model files missing or corrupted")

# -------------------- ALERT MAPPINGS --------------------
alert_to_class = {
    "green": 0,
    "yellow": 1,
    "orange": 2,
    "red": 3
}
class_to_alert = {v: k for k, v in alert_to_class.items()}

ALERT_COLORS = {
    "green": "#10b981",
    "yellow": "#f59e0b",
    "orange": "#f97316",
    "red": "#ef4444"
}

# =========================================================
#                     ROUTES
# =========================================================

# -------- HOME / LANDING --------
@app.route("/")
def index():
    return render_template("index.html")

# -------- LOGIN --------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        # Demo login (replace with DB later)
        if username == "demo" and password == "demo123":
            session["logged_in"] = True
            session["username"] = username
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials", "danger")

    return render_template("login.html")

# -------- SIGNUP --------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        flash("Signup successful! Please login.", "success")
        return redirect(url_for("login"))
    return render_template("signup.html")

# -------- DASHBOARD --------
@app.route("/dashboard")
def dashboard():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("dashboard.html", username=session.get("username"))

# -------- FREE PREDICTION PAGE --------
@app.route("/free_prediction")
def free_prediction():
    return render_template("free_prediction.html")

# -------- PREDICT PAGE --------
@app.route("/predict")
def predict():
    return render_template("predict.html")

# -------- MAKE PREDICTION --------
@app.route("/make_prediction", methods=["POST"])
def make_prediction():
    try:
        magnitude = float(request.form["magnitude"])
        depth = float(request.form["depth"])
        cdi = float(request.form["cdi"])
        mmi = float(request.form["mmi"])
        sig = float(request.form["sig"])

        input_df = pd.DataFrame([{
            "magnitude": magnitude,
            "depth": depth,
            "cdi": cdi,
            "mmi": mmi,
            "sig": sig
        }])

        input_df = input_df[feature_names]
        scaled_input = scaler.transform(input_df)

        predicted_class = model.predict(scaled_input)[0]
        probabilities = model.predict_proba(scaled_input)[0]

        predicted_alert = class_to_alert[predicted_class]

        # ---------- Probability Plot ----------
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = [class_to_alert[i].capitalize() for i in range(len(probabilities))]
        colors = [ALERT_COLORS[class_to_alert[i]] for i in range(len(probabilities))]

        ax.bar(labels, probabilities, color=colors)
        ax.set_ylim(0, 1)
        ax.set_title("Prediction Probabilities")

        for i, p in enumerate(probabilities):
            ax.text(i, p + 0.02, f"{p*100:.1f}%", ha="center")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        plot_url = base64.b64encode(buf.getvalue()).decode()

        return render_template(
            "result.html",
            alert=predicted_alert.upper(),
            color=ALERT_COLORS[predicted_alert],
            plot_url=plot_url,
            time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            prediction_id=str(uuid.uuid4())[:8]
        )

    except Exception as e:
        return render_template("result.html", error=str(e))

# -------- LOGOUT --------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

# -------------------- LOCAL RUN --------------------
if __name__ == "__main__":
    app.run(debug=True)
