import os
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Cloudflare model server base URL (set in Render Environment Variables)
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

    # basic validation (optional but helpful)
    if not text:
        return jsonify({"error": "text is required"}), 400

    if not MODEL_SERVER_URL:
        return jsonify({"error": "MODEL_SERVER_URL not set in Render env"}), 500

    try:
        # Forward to model server
        r = requests.post(
            f"{MODEL_SERVER_URL}/predict",
            json={"text": text},   # ensure format consistent
            timeout=60
        )

        # Return exactly what model server returns
        return (r.text, r.status_code, {"Content-Type": "application/json"})

    except requests.exceptions.Timeout:
        return jsonify({"error": "Model server timeout (check Colab/tunnel)"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # for local testing; Render uses its own server command
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
