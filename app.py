import os
import random
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
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    lower = text.lower()

    political = 1 if any(k in lower for k in ["kerajaan", "parlimen", "undi", "menteri", "politik", "parti"]) else 0
    political_conf = round(random.uniform(0.70, 0.95), 3)

    if any(k in lower for k in ["kerajaan", "menteri", "pm"]):
        target = "gov"
    elif any(k in lower for k in ["parti", "penyokong"]):
        target = "political_group"
    elif any(k in lower for k in ["dia", "individu", "orang"]):
        target = "person"
    else:
        target = "none"
    target_conf = round(random.uniform(0.60, 0.93), 3)

    hate = "hate" if any(k in lower for k in ["benci", "bodoh", "bangsat", "kepala butoh"]) else "none"
    hate_conf = round(random.uniform(0.70, 0.95), 3)

    emotions = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
    emotion = "anger" if hate == "hate" else random.choice(emotions)
    emotion_conf = round(random.uniform(0.60, 0.90), 3)

    return jsonify({
        "input": {"text": text},
        "predictions": {
            "political_topic": {"label": political, "confidence": political_conf},
            "target": {"label": target, "confidence": target_conf},
            "hate_speech": {"label": hate, "confidence": hate_conf},
            "emotion": {"label": emotion, "confidence": emotion_conf}
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
