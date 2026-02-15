import os
import io
import base64
import pickle
import re
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from src.preprocess import TextPreprocessor
except Exception:
    TextPreprocessor = None

app = Flask(__name__)
CORS(app)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

model_store = {
    "preprocessor": None,
    "vectorizer": None,
    "model": None,
    "model_name": None,
}


class BasicPreprocessor:
    """Minimal preprocessor used when trained artifacts are unavailable."""

    @staticmethod
    def preprocess(text: str) -> str:
        text = re.sub(r"https?://\S+|www\.\S+", "", str(text).lower())
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()


def _normalize_sentiment(value):
    label = str(value).strip().lower()
    if label in {"pos", "positive", "1"}:
        return "positive"
    if label in {"neg", "negative", "-1"}:
        return "negative"
    if label in {"neu", "neutral", "0"}:
        return "neutral"
    return "neutral"


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def load_artifacts():
    """Load optional model artifacts; app still runs if none are found."""
    MODELS_DIR.mkdir(exist_ok=True)

    # Preprocessor
    preprocessor_path = MODELS_DIR / "preprocessor.pkl"
    if preprocessor_path.exists():
        try:
            model_store["preprocessor"] = _load_pickle(preprocessor_path)
        except Exception:
            model_store["preprocessor"] = None

    if model_store["preprocessor"] is None:
        if TextPreprocessor is not None:
            try:
                model_store["preprocessor"] = TextPreprocessor()
            except Exception:
                model_store["preprocessor"] = BasicPreprocessor()
        else:
            model_store["preprocessor"] = BasicPreprocessor()

    # Vectorizer
    for name in ("feature_extractor.pkl", "vectorizer.pkl"):
        path = MODELS_DIR / name
        if path.exists():
            try:
                model_store["vectorizer"] = _load_pickle(path)
                break
            except Exception:
                continue

    # Model
    excluded = {
        "preprocessor.pkl",
        "feature_extractor.pkl",
        "vectorizer.pkl",
        "lstm_tokenizer.pkl",
    }

    candidates = [p for p in MODELS_DIR.glob("*.pkl") if p.name not in excluded]

    if candidates:
        try:
            model_path = sorted(candidates)[0]
            model_store["model"] = _load_pickle(model_path)
            model_store["model_name"] = model_path.stem
        except Exception:
            model_store["model"] = None
            model_store["model_name"] = None


def create_pie_chart(sentiment_counts):
    fig, ax = plt.subplots(figsize=(7, 7))

    labels = ["positive", "neutral", "negative"]
    values = [sentiment_counts.get(k, 0) for k in labels]

    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    plt.close(fig)

    return base64.b64encode(buf.read()).decode("utf-8")


def create_bar_chart(sentiment_counts):
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = ["positive", "neutral", "negative"]
    values = [sentiment_counts.get(k, 0) for k in labels]

    bars = ax.bar(labels, values)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, f"{int(h)}", ha="center", va="bottom")

    ax.set_ylabel("Count")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    plt.close(fig)

    return base64.b64encode(buf.read()).decode("utf-8")


def keyword_fallback(text):
    positive_words = {"love", "amazing", "great", "excellent", "good", "best", "awesome", "happy", "wonderful"}
    negative_words = {"hate", "terrible", "worst", "bad", "awful", "disappointed", "angry", "sad", "horrible"}

    tokens = set(str(text).lower().split())
    pos = len(tokens & positive_words)
    neg = len(tokens & negative_words)

    if pos > neg:
        return "positive", min(0.5 + pos * 0.1, 0.95)
    if neg > pos:
        return "negative", min(0.5 + neg * 0.1, 0.95)
    return "neutral", 0.6


def predict_sentiment(text):
    preprocessor = model_store["preprocessor"]

    processed = (
        preprocessor.preprocess(text)
        if hasattr(preprocessor, "preprocess")
        else str(text)
    )

    model = model_store["model"]
    vectorizer = model_store["vectorizer"]

    if model is not None and vectorizer is not None and hasattr(vectorizer, "transform"):
        try:
            features = vectorizer.transform([processed])
            pred = model.predict(features)[0]
            sentiment = _normalize_sentiment(pred)

            if hasattr(model, "predict_proba"):
                confidence = float(np.max(model.predict_proba(features)[0]))
            else:
                confidence = 0.8

            return sentiment, confidence
        except Exception:
            pass

    return keyword_fallback(processed)


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": model_store["model"] is not None,
            "model_name": model_store["model_name"],
            "models_dir": str(MODELS_DIR),
        }
    )


@app.post("/analyze")
def analyze():
    payload = request.get_json(silent=True) or {}
    text_blob = str(payload.get("text", ""))
    texts = [t.strip() for t in text_blob.splitlines() if t.strip()]

    if not texts:
        return jsonify({"error": "No text provided"}), 400

    results = []
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}

    for text in texts:
        sentiment, confidence = predict_sentiment(text)
        sentiment = _normalize_sentiment(sentiment)

        results.append(
            {
                "text": text,
                "sentiment": sentiment,
                "confidence": float(confidence),
            }
        )

        sentiment_counts[sentiment] += 1

    return jsonify(
        {
            "results": results,
            "sentiment_counts": sentiment_counts,
            "pie_chart": create_pie_chart(sentiment_counts),
            "bar_chart": create_bar_chart(sentiment_counts),
            "positive_count": sentiment_counts["positive"],
            "neutral_count": sentiment_counts["neutral"],
            "negative_count": sentiment_counts["negative"],
        }
    )


if __name__ == "__main__":
    load_artifacts()

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    app.run(host=host, port=port, debug=debug)