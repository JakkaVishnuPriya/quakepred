from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import os

# -------------------- APP INIT --------------------
app = Flask(__name__)

# -------------------- ABSOLUTE PATH FIX (RENDER SAFE) --------------------
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
    "green": "#28a745",
    "yellow": "#ffc107",
    "orange": "#fd7e14",
    "red": "#dc3545"
}

# -------------------- ROUTES --------------------
@app.route('/')
def index():
    return render_template('index.html')

# -------------------- LOCAL RUN ONLY --------------------
if __name__ == "__main__":
    app.run(debug=True)
