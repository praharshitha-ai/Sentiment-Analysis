__version__ = "1.0.2"
__author__ = "Sentiment Analysis Team"
__description__ = "Sentiment Analysis source modules"

from .data_loader import DataLoader
from .preprocess import TextPreprocessor, download_nltk_data
from .features import (
    FeatureExtractor,
    SimpleFeatureExtractor,
    AdvancedFeatureExtractor,
    TextFeatureEngineer,
)

__all__ = [
    "DataLoader",
    "TextPreprocessor",
    "download_nltk_data",
    "FeatureExtractor",
    "SimpleFeatureExtractor",
    "AdvancedFeatureExtractor",
    "TextFeatureEngineer",
    "TraditionalModels",
    "LSTMModel",
]


def __getattr__(name):
    """Lazy-load heavy model imports so base package import stays lightweight."""
    if name in {"TraditionalModels", "LSTMModel"}:
        from .models import TraditionalModels, LSTMModel

        return {"TraditionalModels": TraditionalModels, "LSTMModel": LSTMModel}[name]
    raise AttributeError(f"module 'src' has no attribute {name!r}")
