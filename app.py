import random

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    # ---- dummy logic (untuk demo UI + report) ----
    # rules sikit supaya nampak "bijak"
    lower = text.lower()

    political = 1 if any(k in lower for k in ["kerajaan", "parlimen", "undi", "menteri", "politik", "parti"]) else 0
    political_conf = round(random.uniform(0.70, 0.95), 3) if political else round(random.uniform(0.55, 0.80), 3)

    # target
    if any(k in lower for k in ["kerajaan", "menteri", "pm", "yab"]):
        target = "gov"
    elif any(k in lower for k in ["dia", "individu", "orang tu"]):
        target = "person"
    elif any(k in lower for k in ["parti", "penyokong", "puak", "gabungan"]):
        target = "political_group"
    else:
        target = "none"
    target_conf = round(random.uniform(0.60, 0.93), 3)

    # hate
    hate = "hate" if any(k in lower for k in ["benci", "hapus", "bunuh", "bodoh", "bangsat"]) else "none"
    hate_conf = round(random.uniform(0.70, 0.95), 3) if hate == "hate" else round(random.uniform(0.60, 0.90), 3)

    # emotion (8 class)
    emotions = ["anger", "anticipation", "disgust", "fear", "happy
    ", "sadness", "surprise", "trust"]
    # simple cue
    if "!" in text:
        emotion = "surprise"
    elif any(k in lower for k in ["marah", "geram", "menyampah"]):
        emotion = "anger"
    elif any(k in lower for k in ["seronok", "gembira", "best"]):
        emotion = "joy"
    elif any(k in lower for k in ["sedih", "kecewa"]):
        emotion = "sadness"
    else:
        emotion = random.choice(emotions)
    emotion_conf = round(random.uniform(0.55, 0.92), 3)

    return jsonify({
        "input": {"text": text},
        "predictions": {
            "political_topic": {"label": political, "confidence": political_conf},
            "target": {"label": target, "confidence": target_conf},
            "hate_speech": {"label": hate, "confidence": hate_conf},
            "emotion": {"label": emotion, "confidence": emotion_conf}
        }
    })
