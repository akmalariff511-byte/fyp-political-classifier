# ===== FLASK APP =====
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def homepage():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify(status="ok")

@app.route("/predict", methods=["POST"])
def predict():
    # ambil JSON dengan selamat
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    # sementara ni kita echo balik dulu
    return jsonify(
        message="model loaded",
        received_text=text
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
