import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from sentiment_analysis.config import MODELS_DIR, get_config
from src.feature_extractor import FeatureExtractor
from src.preprocessor import TextPreprocessor

SENTIMENT_LABELS = ("positive", "neutral", "negative")


def normalize_sentiment(value) -> str:
    token = str(value).strip().lower()
    if token in {"pos", "positive", "1"}:
        return "positive"
    if token in {"neu", "neutral", "0"}:
        return "neutral"
    if token in {"neg", "negative", "-1"}:
        return "negative"
    return "neutral"


class SentimentPredictor:
    def __init__(self, model_type: str = "best"):
        self.config = get_config()
        self.requested_model_type = model_type
        self.model_type = model_type
        self.preprocessor = None
        self.extractor = None
        self.model = None
        self.models_dir = Path(MODELS_DIR)
        self._load_models()

    @classmethod
    def available_model_types(cls) -> List[str]:
        models_dir = Path(MODELS_DIR)
        if not models_dir.exists():
            return ["best"]

        excluded = {"preprocessor.pkl", "feature_extractor.pkl", "vectorizer.pkl", "lstm_tokenizer.pkl"}
        models = sorted([p.stem for p in models_dir.glob("*.pkl") if p.name not in excluded])
        filtered = [name for name in models if name not in {"best"}]
        return ["best"] + filtered if filtered else ["best"]

    def _resolve_best_model_path(self):
        report_path = self.models_dir / "training_report.json"
        if report_path.exists():
            try:
                with report_path.open("r", encoding="utf-8") as f:
                    report = json.load(f)
                report_best = str(report.get("best_model", "")).strip()
                if report_best:
                    candidate = self.models_dir / f"{report_best}.pkl"
                    if candidate.exists():
                        return candidate
            except Exception:
                pass

        explicit_best = self.models_dir / "best_model.pkl"
        if explicit_best.exists():
            return explicit_best

        fallback_order = ["svm.pkl", "logistic_regression.pkl", "naive_bayes.pkl"]
        for name in fallback_order:
            path = self.models_dir / name
            if path.exists():
                return path

        excluded = {"preprocessor.pkl", "feature_extractor.pkl", "vectorizer.pkl", "lstm_tokenizer.pkl"}
        candidates = sorted([p for p in self.models_dir.glob("*.pkl") if p.name not in excluded])
        return candidates[0] if candidates else None

    def _load_models(self):
        preprocessor_path = self.models_dir / self.config["output"]["preprocessor_file"]
        if preprocessor_path.exists():
            with open(preprocessor_path, "rb") as f:
                self.preprocessor = pickle.load(f)
        else:
            self.preprocessor = TextPreprocessor()

        extractor_path = self.models_dir / self.config["output"]["extractor_file"]
        self.extractor = FeatureExtractor(method="tfidf")
        if extractor_path.exists():
            self.extractor.load(str(extractor_path))
        else:
            # Leave unfitted for now; fallback prediction will be used.
            self.extractor = None

        if self.requested_model_type == "best":
            model_path = self._resolve_best_model_path()
        else:
            model_path = self.models_dir / f"{self.requested_model_type}.pkl"

        if model_path and model_path.exists():
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            self.model_type = model_path.stem

    def _fallback_predict(self, text: str) -> Tuple[str, float]:
        positive_words = {"love", "great", "amazing", "good", "best", "awesome"}
        negative_words = {"hate", "bad", "terrible", "awful", "worst", "horrible"}
        tokens = set(text.lower().split())
        pos = len(tokens & positive_words)
        neg = len(tokens & negative_words)
        if pos > neg:
            return "positive", min(0.5 + pos * 0.1, 0.95)
        if neg > pos:
            return "negative", min(0.5 + neg * 0.1, 0.95)
        return "neutral", 0.6

    def _scores_from_probability(self, row: np.ndarray, classes: np.ndarray) -> Dict[str, float]:
        scores = {label: 0.0 for label in SENTIMENT_LABELS}
        for cls, value in zip(classes, row):
            scores[normalize_sentiment(cls)] = float(value)
        return scores

    def _vader_signal(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        if hasattr(self.preprocessor, "get_sentiment_scores"):
            try:
                scores = self.preprocessor.get_sentiment_scores(text)
                compound = float(scores.get("compound", 0.0))
                if compound >= 0.25:
                    return "positive", min(0.5 + abs(compound), 0.95), scores
                if compound <= -0.25:
                    return "negative", min(0.5 + abs(compound), 0.95), scores
                return "neutral", 0.60, scores
            except Exception:
                pass

        sentiment, confidence = self._fallback_predict(text)
        return sentiment, confidence, {"pos": 0.0, "neu": 0.0, "neg": 0.0, "compound": 0.0}

    def _predict_with_model(self, processed: str):
        features = self.extractor.transform([processed])
        raw_sentiment = self.model.predict(features)[0]
        sentiment = normalize_sentiment(raw_sentiment)

        scores = {label: 0.0 for label in SENTIMENT_LABELS}
        confidence = 0.80

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features)[0]
            classes = getattr(self.model, "classes_", np.array(SENTIMENT_LABELS))
            scores = self._scores_from_probability(probs, classes)
            confidence = float(max(scores.values()))
        elif hasattr(self.model, "decision_function"):
            decision = self.model.decision_function(features)
            if len(np.shape(decision)) == 1:
                decision = np.column_stack([-decision, decision])
            exp_d = np.exp(decision - np.max(decision, axis=1, keepdims=True))
            probs = exp_d / np.sum(exp_d, axis=1, keepdims=True)
            classes = getattr(self.model, "classes_", np.array(SENTIMENT_LABELS[: probs.shape[1]]))
            scores = self._scores_from_probability(probs[0], classes)
            confidence = float(max(scores.values()))
        else:
            scores[sentiment] = confidence

        return sentiment, confidence, scores

    def _calibrate_sentiment(
        self,
        model_sentiment: str,
        model_confidence: float,
        model_scores: Dict[str, float],
        processed: str,
    ):
        vader_sentiment, vader_confidence, vader_raw = self._vader_signal(processed)
        final_sentiment = model_sentiment
        final_confidence = model_confidence
        source = "model"

        compound = float(vader_raw.get("compound", 0.0))
        strong_negative = compound <= -0.35 and float(vader_raw.get("neg", 0.0)) >= 0.25
        strong_positive = compound >= 0.35 and float(vader_raw.get("pos", 0.0)) >= 0.25
        neutral_tokens = {"okay", "ok", "fine", "average", "normal", "expected"}
        token_set = set(str(processed).split())
        has_neutral_cue = len(token_set & neutral_tokens) > 0
        sorted_scores = sorted(model_scores.values(), reverse=True)
        top_score = sorted_scores[0] if sorted_scores else 0.0
        second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        score_gap = top_score - second_score

        # Borderline model output + weak lexical polarity -> neutral
        if (model_confidence < 0.62 or score_gap < 0.12) and abs(compound) < 0.25:
            final_sentiment = "neutral"
            final_confidence = max(model_confidence, 0.60)
            source = "neutral_calibration"
        elif has_neutral_cue and model_confidence < 0.75 and abs(compound) < 0.35:
            final_sentiment = "neutral"
            final_confidence = max(model_confidence, 0.60)
            source = "neutral_cue_calibration"
        elif model_confidence < 0.58 and abs(compound) < 0.15:
            final_sentiment = "neutral"
            final_confidence = max(model_confidence, 0.60)
            source = "neutral_calibration"
        elif model_confidence < 0.55 and vader_sentiment != "neutral":
            final_sentiment = vader_sentiment
            final_confidence = max(model_confidence, vader_confidence)
            source = "lexical_calibration"
        elif model_sentiment == "positive" and strong_negative:
            final_sentiment = "negative"
            final_confidence = max(model_confidence, vader_confidence)
            source = "lexical_override"
        elif model_sentiment == "negative" and strong_positive:
            final_sentiment = "positive"
            final_confidence = max(model_confidence, vader_confidence)
            source = "lexical_override"

        if final_sentiment not in model_scores:
            model_scores[final_sentiment] = final_confidence

        return final_sentiment, float(final_confidence), source, vader_raw, model_scores

    def predict(self, text: str):
        processed = self.preprocessor.preprocess(text)

        if self.model is not None and self.extractor is not None:
            model_sentiment, model_confidence, model_scores = self._predict_with_model(processed)
            sentiment, confidence, source, vader_raw, model_scores = self._calibrate_sentiment(
                model_sentiment,
                model_confidence,
                model_scores,
                processed,
            )
        else:
            sentiment, confidence = self._fallback_predict(processed)
            model_scores = {label: 0.0 for label in SENTIMENT_LABELS}
            model_scores[sentiment] = float(confidence)
            source = "keyword_fallback"
            _, _, vader_raw = self._vader_signal(processed)

        return {
            "text": text,
            "processed_text": processed,
            "sentiment": normalize_sentiment(sentiment),
            "confidence": float(confidence),
            "model": self.model_type,
            "source": source,
            "score_positive": float(model_scores.get("positive", 0.0)),
            "score_neutral": float(model_scores.get("neutral", 0.0)),
            "score_negative": float(model_scores.get("negative", 0.0)),
            "vader_compound": float(vader_raw.get("compound", 0.0)),
        }

    def predict_batch(self, texts):
        return [self.predict(t) for t in texts]

    def analyze_file(self, filepath: str, text_column: str = "text", output_path: str = None):
        path = Path(filepath)
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")

        preds = self.predict_batch(df[text_column].astype(str).tolist())
        df["predicted_sentiment"] = [p["sentiment"] for p in preds]
        df["confidence"] = [p["confidence"] for p in preds]

        if output_path:
            out = Path(output_path)
            if out.suffix.lower() == ".csv":
                df.to_csv(out, index=False)
            elif out.suffix.lower() in {".xlsx", ".xls"}:
                df.to_excel(out, index=False)
        return df


def main():
    parser = argparse.ArgumentParser(description="Sentiment prediction")
    parser.add_argument("-t", "--text", nargs="+", help="Text(s) to analyze")
    parser.add_argument("-f", "--file", help="CSV/Excel file path")
    parser.add_argument("-c", "--column", default="text", help="Text column name")
    parser.add_argument("-o", "--output", help="Optional output file path")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args()

    predictor = SentimentPredictor()

    if args.text:
        results = predictor.predict_batch(args.text)
        if args.format == "json":
            print(json.dumps(results if len(results) > 1 else results[0], indent=2))
        else:
            for r in results:
                print(f"Text: {r['text']}")
                print(f"Sentiment: {r['sentiment']} | Confidence: {r['confidence']:.3f}")
                print("-")
    elif args.file:
        df = predictor.analyze_file(args.file, args.column, args.output)
        print(df.head().to_string())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
