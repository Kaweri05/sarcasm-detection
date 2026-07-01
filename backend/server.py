# c:/Users/kawer/OneDrive/Desktop/sarcasm-detection/backend/server.py
"""
Flask backend for the Sarcasm Detection Dashboard (enhanced version).

Features added:
- Load model accuracy from ``model_metrics.json``.
- Persist each prediction to ``analysis_history.json``.
- Return ``trendAlert`` flag when a high‑confidence sarcastic sentence is detected.
- Expose ``/history`` endpoint for paginated analysis logs.
"""

import json
import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, render_template
from textblob import TextBlob

# ---------------------------------------------------------------------------
# Configuration & Globals
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
app = Flask(
    __name__,
    static_folder=str(BASE_DIR / "static"),
    template_folder=str(BASE_DIR / "templates"),
)

# Load or initialise model metrics (accuracy). Stored as JSON: {"accuracy": <float>}
METRICS_PATH = BASE_DIR / "backend" / "model_metrics.json"
if METRICS_PATH.exists():
    with METRICS_PATH.open() as f:
        model_metrics = json.load(f)
else:
    model_metrics = {"accuracy": 78.34}  # fallback value

# History file – a simple list of prediction records
HISTORY_PATH = BASE_DIR / "backend" / "analysis_history.json"
_history_lock = threading.Lock()
if not HISTORY_PATH.exists():
    with HISTORY_PATH.open("w") as f:
        json.dump([], f)

# ---------------------------------------------------------------------------
# Helper utilities (language detection & sentiment)
# ---------------------------------------------------------------------------
HIRDI_WORDS = ["kya", "hai", "fir", "waah", "wah", "acha", "kyu", "nahi"]
MARATHI_WORDS = [
    "kay",
    "kaay",
    "kasa",
    "aahe",
    "ahe",
    "khup",
    "mast",
    "punha",
    "bara",
    "chan",
    "nahi",
    "zala",
    "zhali",
    "mhanje",
    "mala",
    "tula",
    "apla",
    "marathi",
]


def detect_language(text: str) -> str:
    """Very naive word‑list based language detection."""
    lowered = text.lower()
    hindi_cnt = sum(word in lowered for word in HIRDI_WORDS)
    marathi_cnt = sum(word in lowered for word in MARATHI_WORDS)
    if marathi_cnt > hindi_cnt and marathi_cnt > 0:
        return "Marathi"
    elif hindi_cnt > 0:
        return "Hindi / Hinglish"
    else:
        return "English"


def analyse_sentiment(text: str) -> str:
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "😊 Positive"
    if polarity < 0:
        return "😔 Negative"
    return "😐 Neutral"


def predict_sarcasm(text: str):
    """Simple heuristic sarcasm detector.

    - Exclamation mark or all‑uppercase → sarcastic (confidence 75).
    - Otherwise → not sarcastic (confidence 25).
    """
    if "!" in text or text.isupper():
        return True, 75.0
    return False, 25.0

# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def append_history(entry: dict):
    """Thread‑safe append of a prediction entry to ``analysis_history.json``."""
    with _history_lock:
        with HISTORY_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data.append(entry)
        with HISTORY_PATH.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    """Render the main dashboard page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Accept JSON ``{ "text": "..." }`` and return prediction details.

    The response now includes:
    * ``accuracy`` – overall model accuracy from ``model_metrics.json``.
    * ``trendAlert`` – ``true`` when the prediction is sarcastic **and** confidence > 80.
    """
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    language = detect_language(text)
    sentiment = analyse_sentiment(text)
    sarcastic, confidence = predict_sarcasm(text)

    response = {
        "language": language,
        "sentiment": sentiment,
        "sarcasm": sarcastic,
        "confidence": round(confidence, 2),
        "accuracy": model_metrics.get("accuracy", 0),
        "trendAlert": sarcastic and confidence > 80,
        "plotData": {
            "labels": ["Sarcastic", "Not Sarcastic"],
            "values": [round(confidence, 2), round(100 - confidence, 2)],
        },
    }

    # Persist analysis record
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "text": text,
        "language": language,
        "sentiment": sentiment,
        "sarcasm": sarcastic,
        "confidence": round(confidence, 2),
    }
    try:
        append_history(entry)
    except Exception as e:
        app.logger.error(f"Failed to write history: {e}")

    return jsonify(response)


@app.route("/history", methods=["GET"])
def history():
    """Return paginated analysis history.

    Query parameters:
    * ``page`` – 1‑based page number (default 1).
    * ``size`` – items per page (default 20, max 100).
    """
    try:
        page = int(request.args.get("page", 1))
        size = int(request.args.get("size", 20))
        size = min(size, 100)
    except ValueError:
        return jsonify({"error": "Invalid pagination parameters"}), 400

    with _history_lock:
        with HISTORY_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    total = len(data)
    start = (page - 1) * size
    end = start + size
    page_items = data[start:end]
    return jsonify({
        "page": page,
        "size": size,
        "total": total,
        "records": page_items,
    })

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
