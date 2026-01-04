import os
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def homepage():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_server": os.environ.get("MODEL_SERVER_URL", "")
    })

@app.route("/predict", methods=["POST"])
def predict():
    MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "").strip()
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not MODEL_SERVER_URL:
        return jsonify({"error": "MODEL_SERVER_URL not set"}), 400

    try:
        r = requests.post(
            MODEL_SERVER_URL.rstrip("/") + "/predict",
            json={"text": text},
            timeout=60
        )
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
