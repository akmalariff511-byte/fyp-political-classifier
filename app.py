import os, zipfile, requests
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModel

app = Flask(__name__)

# ---------------------------
# 1) Download & extract model
# ---------------------------
MODEL_DIR = "xlmr_multitask_701020"
MODEL_ZIP_URL = os.environ.get("MODEL_ZIP_URL", "").strip()

def ensure_model():
    if os.path.isdir(MODEL_DIR) and os.path.exists(os.path.join(MODEL_DIR, "pytorch_model_multitask.pt")):
        print("✅ Model already present")
        return

    if not MODEL_ZIP_URL:
        raise RuntimeError("MODEL_ZIP_URL not set")

    print("⬇️ Downloading model zip...")
    r = requests.get(MODEL_ZIP_URL, stream=True, timeout=600)
    r.raise_for_status()

    with open("model.zip", "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    print("📦 Extracting model...")
    with zipfile.ZipFile("model.zip", "r") as z:
        z.extractall(".")

    print("✅ Model ready")

ensure_model()

# ---------------------------
# 2) Define multitask model
# ---------------------------
MODEL_NAME = "xlm-roberta-base"
device = torch.device("cpu")  # Render free usually CPU

class XLMRMultiTask(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        h = self.encoder.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.head_politic = nn.Linear(h, 1)   # BCE
        self.head_target  = nn.Linear(h, 4)   # CE
        self.head_hate    = nn.Linear(h, 1)   # BCE
        self.head_emotion = nn.Linear(h, 8)   # CE

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        x = self.drop(cls)
        return (
            self.head_politic(x).squeeze(-1),
            self.head_target(x),
            self.head_hate(x).squeeze(-1),
            self.head_emotion(x),
        )

# load tokenizer + model weights
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = XLMRMultiTask().to(device)
state = torch.load(os.path.join(MODEL_DIR, "pytorch_model_multitask.pt"), map_location=device)
model.load_state_dict(state)
model.eval()

# label maps
target_names  = ["no targeted", "person", "gov", "political group"]
emotion_names = ["anger","anticipation","disgust","fear","happy","sadness","surprise","trust"]

def run_inference(text: str):
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    with torch.no_grad():
        log_pol, log_tgt, log_hate, log_emo = model(input_ids, attn)

        pol_prob  = torch.sigmoid(log_pol).item()
        hate_prob = torch.sigmoid(log_hate).item()

        tgt_id = int(torch.argmax(log_tgt, dim=1).item())
        emo_id = int(torch.argmax(log_emo, dim=1).item())

    # thresholds
    pol_label  = "political" if pol_prob >= 0.5 else "non-political"
    hate_label = "hate" if hate_prob >= 0.5 else "none"

    return {
        "politic": {"label": pol_label, "score": float(pol_prob)},
        "target": {"label": target_names[tgt_id], "id": tgt_id},
        "hate": {"label": hate_label, "score": float(hate_prob)},
        "emotion": {"label": emotion_names[emo_id], "id": emo_id},
    }

# ---------------------------
# 3) Routes
# ---------------------------
@app.route("/")
def homepage():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "model_zip_url_set": bool(MODEL_ZIP_URL),
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    try:
        return jsonify(run_inference(text))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
