__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "Sentiment Analysis AI - Source Modules"

# Import main classes for easy access
from .data_loader import DataLoader
from .preprocessor import TextPreprocessor
from .feature_extractor import FeatureExtractor
from .models import TraditionalModels, LSTMModel

# Define what gets imported with "from src import *"
__all__ = [
    'DataLoader',
    'TextPreprocessor', 
    'FeatureExtractor',
    'TraditionalModels',
    'LSTMModel',
    'get_config',
    'download_nltk_data'
]

# Package-level utilities
import os
import sys
from pathlib import Path

def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent.absolute()

def get_data_path():
    """Get the data directory path"""
    return get_project_root() / "data"

def get_models_path():
    """Get the models directory path"""
    return get_project_root() / "models"

def ensure_directories():
    """Ensure all necessary directories exist"""
    dirs = ['data', 'models', 'logs', 'app', 'notebooks']
    root = get_project_root()
    for d in dirs:
        (root / d).mkdir(exist_ok=True)

# Auto-create directories on import
ensure_directories()

print(f"✅ Sentiment Analysis Package v{__version__} loaded")