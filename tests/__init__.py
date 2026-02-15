import os
import sys
from pathlib import Path

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

# Test configuration
TEST_CONFIG = {
    'test_data_size': 100,
    'random_seed': 42,
    'tolerance': 0.01,  # For floating point comparisons
    'timeout': 30  # seconds for long-running tests
}

# Test data samples
SAMPLE_TEXTS = {
    'positive': [
        "I absolutely love this product! It's amazing!",
        "Best purchase ever! Highly recommend to everyone!",
        "Fantastic quality and great customer service!",
        "Wonderful experience, will definitely buy again!",
        "Outstanding performance, exceeded my expectations!"
    ],
    'negative': [
        "Terrible product, complete waste of money!",
        "Worst experience ever, never buying again!",
        "Horrible quality, broke after one day!",
        "Very disappointed with this purchase!",
        "Awful service, rude staff, never again!"
    ],
    'neutral': [
        "The product is okay, nothing special.",
        "It's fine, does what it's supposed to do.",
        "Average quality, acceptable for the price.",
        "Not bad, but could be better.",
        "Decent product, meets basic requirements."
    ],
    'edge_cases': [
        "",  # Empty string
        "!!!???",  # Only punctuation
        "12345",  # Only numbers
        "http://example.com @user #hashtag",  # Only special tokens
        "Good bad good bad",  # Mixed sentiment
        "VERY GOOD!!!",  # Caps and intensifiers
        "Not bad, not good, just okay..."  # Complex negation
    ]
}

def get_test_data():
    """Return test data dictionary"""
    return SAMPLE_TEXTS.copy()

def get_sample_dataframe():
    """Create sample dataframe for testing"""
    import pandas as pd
    
    data = []
    for sentiment, texts in SAMPLE_TEXTS.items():
        if sentiment != 'edge_cases':
            for text in texts:
                data.append({'text': text, 'sentiment': sentiment})
    
    return pd.DataFrame(data)

# Skip slow tests decorator
import unittest

def skip_slow_tests():
    """Decorator to skip slow tests"""
    return unittest.skipIf(
        os.environ.get('SKIP_SLOW_TESTS', 'False').lower() == 'true',
        "Slow test skipped"
    )

def require_model_files():
    """Skip if model files don't exist"""
    models_dir = project_root / "models"
    has_models = any(models_dir.glob("*.pkl")) if models_dir.exists() else False
    return unittest.skipUnless(has_models, "Model files not found")

__all__ = [
    'TEST_CONFIG',
    'SAMPLE_TEXTS',
    'get_test_data',
    'get_sample_dataframe',
    'skip_slow_tests',
    'require_model_files'
]