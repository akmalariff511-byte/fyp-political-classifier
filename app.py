# ===== FLASK APP =====
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


@app.route("/")
def homepage():
    return render_template("index.html")

@app.route("/health")
def health():
    return {"status": "ok"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    return {"message": "model loaded", "input": data}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
