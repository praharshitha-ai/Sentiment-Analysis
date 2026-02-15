import re
import unicodedata
from collections import Counter
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import TweetTokenizer


def download_nltk_data() -> None:
    """Download required NLTK resources if missing."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("sentiment/vader_lexicon", "vader_lexicon"),
    ]
    for path_key, package in resources:
        try:
            nltk.data.find(path_key)
        except LookupError:
            nltk.download(package, quiet=True)


class TextPreprocessor:
    """Text preprocessing utility for sentiment tasks."""

    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        remove_hashtags: bool = False,
        remove_emojis: bool = False,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        remove_stopwords: bool = False,
        custom_stopwords: Optional[Iterable[str]] = None,
        lemmatize: bool = True,
        stem: bool = False,
        stemmer: str = "wordnet",
        min_word_length: int = 2,
        max_word_length: int = 20,
        language: str = "english",
        preserve_case_for_emojis: bool = True,
    ):
        self.config = {
            "lowercase": lowercase,
            "remove_urls": remove_urls,
            "remove_mentions": remove_mentions,
            "remove_hashtags": remove_hashtags,
            "remove_emojis": remove_emojis,
            "remove_punctuation": remove_punctuation,
            "remove_numbers": remove_numbers,
            "remove_stopwords": remove_stopwords,
            "lemmatize": lemmatize,
            "stem": stem,
            "min_word_length": min_word_length,
            "max_word_length": max_word_length,
        }

        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self.mention_pattern = re.compile(r"@\w+")
        self.hashtag_pattern = re.compile(r"#(\w+)")
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "]+",
            flags=re.UNICODE,
        )

        self.tweet_tokenizer = TweetTokenizer(
            preserve_case=not lowercase,
            reduce_len=True,
            strip_handles=remove_mentions,
        )

        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = self._get_stemmer(stemmer, language)

        try:
            self.stop_words = set(stopwords.words(language))
        except LookupError:
            download_nltk_data()
            self.stop_words = set(stopwords.words(language))

        if custom_stopwords:
            self.stop_words.update(custom_stopwords)

        self.stop_words.update({"rt", "via", "amp", "http", "https", "co", "www"})

        try:
            self.sia = SentimentIntensityAnalyzer()
        except LookupError:
            download_nltk_data()
            self.sia = SentimentIntensityAnalyzer()

    def _get_stemmer(self, stemmer_type: str, language: str):
        if stemmer_type == "porter":
            return PorterStemmer()
        if stemmer_type == "snowball":
            return SnowballStemmer(language)
        return None

    def clean_text(self, text: str) -> str:
        text = str(text)
        text = unicodedata.normalize("NFKD", text)

        if self.config["lowercase"]:
            text = text.lower()

        if self.config["remove_urls"]:
            text = self.url_pattern.sub(" ", text)

        if self.config["remove_mentions"]:
            text = self.mention_pattern.sub(" ", text)

        if self.config["remove_hashtags"]:
            text = self.hashtag_pattern.sub(r"\1", text)
        else:
            text = text.replace("#", "")

        if self.config["remove_emojis"]:
            text = self.emoji_pattern.sub(" ", text)

        if self.config["remove_numbers"]:
            text = re.sub(r"\d+", " ", text)

        if self.config["remove_punctuation"]:
            text = re.sub(r"[^\w\s]", " ", text)

        text = re.sub(r"\s+", " ", text).strip()
        return text

    def preprocess(self, text: str) -> str:
        cleaned = self.clean_text(text)
        if not cleaned:
            return ""

        tokens = self.tweet_tokenizer.tokenize(cleaned)

        processed = []
        for token in tokens:
            if self.config["remove_stopwords"] and token in self.stop_words:
                continue
            if not (self.config["min_word_length"] <= len(token) <= self.config["max_word_length"]):
                continue

            if self.config["lemmatize"]:
                token = self.lemmatizer.lemmatize(token)
            if self.config["stem"] and self.stemmer is not None:
                token = self.stemmer.stem(token)

            processed.append(token)

        return " ".join(processed)

    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in dataframe")

        out = df.copy()
        out["processed_text"] = out[text_column].fillna("").astype(str).apply(self.preprocess)
        return out

    def get_sentiment_scores(self, text: str) -> Dict[str, float]:
        return self.sia.polarity_scores(str(text))

    def get_word_stats(self, df: pd.DataFrame, processed_column: str = "processed_text") -> Dict[str, float]:
        if processed_column not in df.columns:
            raise ValueError(f"Column '{processed_column}' not found in dataframe")

        texts = df[processed_column].fillna("").astype(str)
        tokenized = [t.split() for t in texts]

        total_words = int(sum(len(tokens) for tokens in tokenized))
        unique_words = int(len(set(word for tokens in tokenized for word in tokens)))
        avg_words = float(total_words / len(tokenized)) if len(tokenized) else 0.0

        freq = Counter(word for tokens in tokenized for word in tokens)

        return {
            "total_words": total_words,
            "unique_words": unique_words,
            "avg_words_per_text": avg_words,
            "top_words": freq.most_common(20),
        }


if __name__ == "__main__":
    sample = "I absolutely LOVE this product!!! https://example.com #awesome"
    p = TextPreprocessor()
    print(p.preprocess(sample))

