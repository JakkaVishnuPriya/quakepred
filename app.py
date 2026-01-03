
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the pre-trained model, scaler, and feature names
try:
    model = joblib.load('earthquake_rf_model.pkl')
    scaler = joblib.load('earthquake_scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    print("Model, scaler, and feature names loaded successfully.")
except FileNotFoundError:
    print("Error: Model or scaler files not found. Please ensure 'earthquake_rf_model.pkl', 'earthquake_scaler.pkl', and 'feature_names.pkl' are in the same directory.")
    # Exit or handle error appropriately
    exit()

# Define alert level mappings
alert_to_class = {
    "green": 0,
    "yellow": 1,
    "orange": 2,
    "red": 3
}
class_to_alert = {v: k for k, v in alert_to_class.items()}

# Define alert colors for UI
ALERT_COLORS = {
    "green": "#28a745",  # Bootstrap green
    "yellow": "#ffc107", # Bootstrap yellow
    "orange": "#fd7e14", # Custom orange for consistency
    "red": "#dc3545"     # Bootstrap red
}

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
