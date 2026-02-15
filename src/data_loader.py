from pathlib import Path
from typing import Optional

import pandas as pd


class DataLoader:
    """Dataset loader for Twitter / IMDB / Amazon sentiment datasets."""

    def _sample_twitter(self) -> pd.DataFrame:
        rows = [
            {"text": "I love this product", "sentiment": "positive"},
            {"text": "Terrible experience and bad quality", "sentiment": "negative"},
            {"text": "It is okay, not great not bad", "sentiment": "neutral"},
        ]
        return pd.DataFrame(rows * 200)

    @staticmethod
    def _read_table(path: Optional[str]) -> Optional[pd.DataFrame]:
        if not path:
            return None

        p = Path(path)
        if not p.exists():
            return None

        suffix = p.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(p)
        if suffix == ".zip":
            return pd.read_csv(p, compression="zip")
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(p)

        raise ValueError(f"Unsupported dataset format: {p.suffix}")

    @staticmethod
    def _normalize_sentiment_labels(df: pd.DataFrame, sentiment_col: str = "sentiment") -> pd.DataFrame:
        out = df.copy()
        if sentiment_col not in out.columns:
            return out

        out[sentiment_col] = out[sentiment_col].astype(str).str.strip().str.lower()
        mapping = {
            "pos": "positive",
            "neg": "negative",
            "neu": "neutral",
            "1": "positive",
            "0": "neutral",
            "-1": "negative",
            "positive": "positive",
            "negative": "negative",
            "neutral": "neutral",
        }
        out[sentiment_col] = out[sentiment_col].replace(mapping)
        return out

    def load_twitter_data(self, path: Optional[str] = None) -> pd.DataFrame:
        df = self._read_table(path)
        if df is None:
            return self._sample_twitter()

        text_candidates = ["text", "tweet", "content"]
        sentiment_candidates = ["sentiment", "label", "airline_sentiment", "target"]

        text_col = next((c for c in text_candidates if c in df.columns), None)
        sentiment_col = next((c for c in sentiment_candidates if c in df.columns), None)

        if text_col is None or sentiment_col is None:
            raise ValueError("Twitter dataset needs text and sentiment columns")

        out = df[[text_col, sentiment_col]].rename(columns={text_col: "text", sentiment_col: "sentiment"})
        return self._normalize_sentiment_labels(out)

    def load_imdb_data(self, path: Optional[str] = None) -> pd.DataFrame:
        df = self._read_table(path)
        if df is None:
            return self._sample_twitter().rename(columns={"text": "review"})

        text_candidates = ["review", "text", "sentence"]
        sentiment_candidates = ["sentiment", "label"]

        text_col = next((c for c in text_candidates if c in df.columns), None)
        sentiment_col = next((c for c in sentiment_candidates if c in df.columns), None)

        if text_col is None or sentiment_col is None:
            raise ValueError("IMDB dataset needs review/text and sentiment columns")

        out = df[[text_col, sentiment_col]].rename(columns={text_col: "review", sentiment_col: "sentiment"})
        return self._normalize_sentiment_labels(out)

    def load_amazon_data(self, path: Optional[str] = None) -> pd.DataFrame:
        df = self._read_table(path)
        if df is None:
            return self._sample_twitter().rename(columns={"text": "review_text"})

        # Supports your zip schema: reviews.text + reviews.rating
        text_candidates = ["reviews.text", "review_text", "review", "text"]
        rating_candidates = ["reviews.rating", "rating", "score", "stars"]
        sentiment_candidates = ["sentiment", "label"]

        text_col = next((c for c in text_candidates if c in df.columns), None)
        if text_col is None:
            raise ValueError("Amazon dataset needs a review text column")

        sentiment_col = next((c for c in sentiment_candidates if c in df.columns), None)
        if sentiment_col is not None:
            out = df[[text_col, sentiment_col]].rename(columns={text_col: "review_text", sentiment_col: "sentiment"})
            return self._normalize_sentiment_labels(out)

        rating_col = next((c for c in rating_candidates if c in df.columns), None)
        if rating_col is None:
            raise ValueError("Amazon dataset needs a rating column when sentiment is not present")

        out = df[[text_col, rating_col]].rename(columns={text_col: "review_text", rating_col: "rating"})
        out["rating"] = pd.to_numeric(out["rating"], errors="coerce")
        out = out.dropna(subset=["rating"]).copy()

        def map_rating_to_sentiment(rating: float) -> str:
            if rating >= 4:
                return "positive"
            if rating <= 2:
                return "negative"
            return "neutral"

        out["sentiment"] = out["rating"].apply(map_rating_to_sentiment)
        return out[["review_text", "sentiment"]]
