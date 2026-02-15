import pickle
from typing import Iterable, List, Optional

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

try:
    from gensim.models import Word2Vec

    GENSIM_AVAILABLE = True
except Exception:
    Word2Vec = None
    GENSIM_AVAILABLE = False


class SimpleFeatureExtractor:
    """Feature extractor supporting TF-IDF, Count, and Word2Vec."""

    def __init__(self, method: str = "tfidf", max_features: int = 5000, ngram_range=(1, 2), vector_size: int = 100):
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vector_size = vector_size

        self.vectorizer = None
        self.word2vec_model = None

    def _build_vectorizer(self):
        if self.method == "count":
            return CountVectorizer(max_features=self.max_features, ngram_range=self.ngram_range)
        return TfidfVectorizer(max_features=self.max_features, ngram_range=self.ngram_range)

    def _texts_to_w2v(self, texts: List[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            tokens = str(text).split()
            token_vecs = [self.word2vec_model.wv[t] for t in tokens if t in self.word2vec_model.wv]
            if token_vecs:
                vectors.append(np.mean(token_vecs, axis=0))
            else:
                vectors.append(np.zeros(self.vector_size, dtype=float))
        return np.array(vectors, dtype=float)

    def fit_transform(self, texts: Iterable[str]):
        texts = list(texts)

        if self.method == "word2vec":
            if not GENSIM_AVAILABLE:
                raise ImportError("Word2Vec requires gensim. Install with: pip install gensim")

            tokenized = [str(t).split() for t in texts]
            self.word2vec_model = Word2Vec(
                sentences=tokenized,
                vector_size=self.vector_size,
                window=5,
                min_count=1,
                workers=4,
                sg=1,
                epochs=10,
            )
            return self._texts_to_w2v(texts)

        self.vectorizer = self._build_vectorizer()
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: Iterable[str]):
        texts = list(texts)

        if self.method == "word2vec":
            if self.word2vec_model is None:
                raise ValueError("Word2Vec extractor is not fitted. Call fit_transform first.")
            return self._texts_to_w2v(texts)

        if self.vectorizer is None:
            raise ValueError("Extractor is not fitted. Call fit_transform first.")
        return self.vectorizer.transform(texts)

    def save(self, filepath: str):
        payload = {
            "method": self.method,
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "vector_size": self.vector_size,
            "vectorizer": self.vectorizer,
            "word2vec_model": self.word2vec_model,
        }
        with open(filepath, "wb") as f:
            pickle.dump(payload, f)

    def load(self, filepath: str):
        with open(filepath, "rb") as f:
            payload = pickle.load(f)

        # Backward compatibility: older files may store just the sklearn vectorizer.
        if isinstance(payload, dict) and "method" in payload:
            self.method = payload.get("method", self.method)
            self.max_features = payload.get("max_features", self.max_features)
            self.ngram_range = tuple(payload.get("ngram_range", self.ngram_range))
            self.vector_size = payload.get("vector_size", self.vector_size)
            self.vectorizer = payload.get("vectorizer")
            self.word2vec_model = payload.get("word2vec_model")
        else:
            self.method = "tfidf"
            self.vectorizer = payload
            self.word2vec_model = None

        return self


class AdvancedFeatureExtractor:
    """Combines TF-IDF with lexicon/statistical features for tests and experiments."""

    def __init__(self, methods: Optional[List[str]] = None, max_features: int = 5000):
        self.methods = methods if methods is not None else ["tfidf", "lexicon", "statistical"]
        self.max_features = max_features
        self.tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2)) if "tfidf" in self.methods else None
        self.fitted = False

    def extract_sentiment_lexicon_features(self, texts: Iterable[str]) -> np.ndarray:
        positive_words = {"good", "great", "excellent", "amazing", "love", "best", "awesome", "happy", "wonderful", "perfect"}
        negative_words = {"bad", "terrible", "awful", "hate", "worst", "horrible", "poor", "sad", "angry", "disappointed"}
        intensifiers = {"very", "really", "extremely", "absolutely", "totally"}

        rows = []
        for text in texts:
            t = str(text)
            words = t.lower().split()
            chars = list(t)
            exclamation_count = t.count("!")
            question_count = t.count("?")

            pos = sum(1 for w in words if w in positive_words)
            neg = sum(1 for w in words if w in negative_words)
            inten = sum(1 for w in words if w in intensifiers)
            upper = sum(1 for c in chars if c.isupper())
            total_words = len(words)
            total_chars = len(chars)

            rows.append([
                pos,
                neg,
                pos - neg,
                (pos + 1) / (neg + 1),
                inten,
                exclamation_count,
                question_count,
                upper / total_chars if total_chars else 0.0,
                total_words,
                np.mean([len(w) for w in words]) if words else 0.0,
            ])

        return np.array(rows, dtype=float)

    def extract_punctuation_features(self, texts: Iterable[str]) -> np.ndarray:
        rows = []
        for text in texts:
            t = str(text)
            rows.append([t.count("!"), t.count("?"), t.count("..."), t.count(","), t.count(".")])
        return np.array(rows, dtype=float)

    def _extract_statistical_features(self, texts: Iterable[str]) -> np.ndarray:
        rows = []
        for text in texts:
            t = str(text)
            words = t.split()
            rows.append([
                len(t),
                len(words),
                len(set(words)) / len(words) if words else 0.0,
                np.mean([len(w) for w in words]) if words else 0.0,
            ])
        return np.array(rows, dtype=float)

    def fit_transform(self, texts: Iterable[str]):
        texts = list(texts)
        matrices = []

        if "tfidf" in self.methods:
            matrices.append(self.tfidf.fit_transform(texts))
        if "lexicon" in self.methods:
            matrices.append(csr_matrix(self.extract_sentiment_lexicon_features(texts)))
        if "statistical" in self.methods:
            matrices.append(csr_matrix(self._extract_statistical_features(texts)))
        if "punctuation" in self.methods:
            matrices.append(csr_matrix(self.extract_punctuation_features(texts)))

        if not matrices:
            result = csr_matrix((len(texts), 0))
        elif len(matrices) == 1:
            result = matrices[0]
        else:
            result = hstack(matrices).tocsr()

        self.fitted = True
        return result

    def transform(self, texts: Iterable[str]):
        if not self.fitted and "tfidf" in self.methods:
            raise ValueError("Extractor is not fitted. Call fit_transform first.")

        texts = list(texts)
        matrices = []

        if "tfidf" in self.methods:
            matrices.append(self.tfidf.transform(texts))
        if "lexicon" in self.methods:
            matrices.append(csr_matrix(self.extract_sentiment_lexicon_features(texts)))
        if "statistical" in self.methods:
            matrices.append(csr_matrix(self._extract_statistical_features(texts)))
        if "punctuation" in self.methods:
            matrices.append(csr_matrix(self.extract_punctuation_features(texts)))

        if not matrices:
            return csr_matrix((len(texts), 0))
        if len(matrices) == 1:
            return matrices[0]
        return hstack(matrices).tocsr()


class TextFeatureEngineer(AdvancedFeatureExtractor):
    """Backward-compatible alias used by existing imports."""


FeatureExtractor = SimpleFeatureExtractor


if __name__ == "__main__":
    sample = ["I love this product!", "Terrible service", "It is okay"]
    ext = SimpleFeatureExtractor(max_features=50)
    mat = ext.fit_transform(sample)
    print(mat.shape)
