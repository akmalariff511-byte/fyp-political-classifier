import os
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "").rstrip("/")

@app.route("/")
def homepage():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify(status="ok", model_server=MODEL_SERVER_URL)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "text is required"}), 400

    if not MODEL_SERVER_URL:
        return jsonify({"error": "MODEL_SERVER_URL not set"}), 500

    r = requests.post(
        f"{MODEL_SERVER_URL}/predict",
        json={"text": text},
        timeout=60
    )

    return (r.text, r.status_code, {"Content-Type": "application/json"})
