import random
from flask import request, jsonify

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    lower = text.lower()

    # Political topic
    political = 1 if any(k in lower for k in ["kerajaan", "parlimen", "undi", "menteri", "politik", "parti"]) else 0
    political_conf = round(random.uniform(0.70, 0.95), 3)

    # Target
    if any(k in lower for k in ["kerajaan", "menteri", "pm"]):
        target = "gov"
    elif any(k in lower for k in ["parti", "penyokong"]):
        target = "political_group"
    elif any(k in lower for k in ["dia", "individu", "orang"]):
        target = "person"
    else:
        target = "none"
    target_conf = round(random.uniform(0.60, 0.93), 3)

    # Hate speech
    hate = "hate" if any(k in lower for k in ["benci", "bodoh", "bangsat", "kepala butoh"]) else "none"
    hate_conf = round(random.uniform(0.70, 0.95), 3)

    # Emotion (8 classes)
    emotions = ["anger", "anticipation", "disgust", "fear", "happy", "sadness", "surprise", "trust"]
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
