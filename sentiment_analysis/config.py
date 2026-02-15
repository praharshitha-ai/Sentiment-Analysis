import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SRC_DIR = BASE_DIR / "src"
APP_DIR = BASE_DIR / "app"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(exist_ok=True)

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

DATA_CONFIG = {
    # Dataset selection: 'twitter', 'imdb', 'amazon', or 'custom'
    'dataset_type': 'twitter',
    
    # File paths (None = use sample data generator)
    'twitter_data_path': None,  # 'data/twitter_sentiment.csv'
    'imdb_data_path': None,     # 'data/imdb_reviews.csv'
    'amazon_data_path': None,   # 'data/amazon_reviews.csv'
    'custom_data_path': None,     # 'data/custom.csv'
    
    # Text column names in CSV files
    'text_columns': {
        'twitter': 'text',
        'imdb': 'review',
        'amazon': 'review_text',
        'custom': 'text'
    },
    
    # Label column names
    'label_columns': {
        'twitter': 'sentiment',
        'imdb': 'sentiment',
        'amazon': 'rating',  # Will be converted to sentiment
        'custom': 'sentiment'
    },
    
    # Sample data size (for auto-generated data)
    'sample_size': 1500,
    
    # Test split ratio
    'test_size': 0.2,
    'random_state': 42
}

# =============================================================================
# PREPROCESSING CONFIGURATION
# =============================================================================

PREPROCESSING_CONFIG = {
    # Text cleaning options
    'remove_urls': True,
    'remove_mentions': True,
    'remove_hashtags': False,  # Keep hashtag words, remove only # symbol
    'remove_emojis': False,    # Keep emojis (they carry sentiment!)
    'remove_punctuation': True,
    'lowercase': True,
    
    # Tokenization
    'tokenization': 'word',  # 'word' or 'sentence'
    
    # Stopwords
    'remove_stopwords': True,
    'custom_stopwords': {'rt', 'via', 'amp', 'http', 'https', 'co', 'www'},
    
    # Normalization
    'lemmatization': True,   # Use WordNet lemmatizer
    'stemming': False,       # Use Porter stemmer (alternative to lemmatization)
    
    # Advanced options
    'min_word_length': 2,    # Remove words shorter than this
    'max_word_length': 20,   # Remove words longer than this (likely noise)
}

# =============================================================================
# FEATURE EXTRACTION CONFIGURATION
# =============================================================================

FEATURE_CONFIG = {
    # Primary method: 'tfidf', 'count', 'word2vec'
    'primary_method': 'tfidf',
    
    # TF-IDF settings
    'tfidf': {
        'max_features': 5000,
        'ngram_range': (1, 2),      # Unigrams + Bigrams
        'min_df': 2,                # Ignore terms in < 2 documents
        'max_df': 0.95,             # Ignore terms in > 95% of documents
        'sublinear_tf': True,       # Use sublinear tf scaling
        'norm': 'l2',               # L2 normalization
        'use_idf': True
    },
    
    # Count Vectorizer settings (alternative)
    'count': {
        'max_features': 5000,
        'ngram_range': (1, 2),
        'min_df': 2,
        'max_df': 0.95
    },
    
    # Word2Vec settings
    'word2vec': {
        'vector_size': 100,
        'window': 5,
        'min_count': 1,
        'workers': 4,
        'sg': 1,                    # 1=Skip-gram, 0=CBOW
        'epochs': 10
    }
}

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODEL_CONFIG = {
    # Traditional ML Models to train
    'train_naive_bayes': True,
    'train_logistic_regression': True,
    'train_svm': True,
    'train_random_forest': False,  # Set True to enable
    
    # Model hyperparameters
    'naive_bayes': {
        'alpha': 1.0,               # Smoothing parameter
        'fit_prior': True
    },
    
    'logistic_regression': {
        'C': 1.0,                   # Regularization strength
        'max_iter': 1000,
        'solver': 'lbfgs',
        'multi_class': 'auto'
    },
    
    'svm': {
        'C': 1.0,
        'max_iter': 1000,
        'penalty': 'l2'
    },
    
    'random_forest': {
        'n_estimators': 100,
        'max_depth': None,
        'random_state': 42
    }
}

# =============================================================================
# LSTM (DEEP LEARNING) CONFIGURATION
# =============================================================================

LSTM_CONFIG = {
    'enabled': True,
    
    # Architecture
    'max_words': 5000,              # Vocabulary size
    'max_len': 50,                  # Maximum sequence length
    'embedding_dim': 128,           # Embedding layer size
    
    # LSTM layers
    'lstm_units_1': 128,
    'lstm_units_2': 64,
    'bidirectional': True,
    'dropout_rate': 0.2,
    'recurrent_dropout': 0.2,
    
    # Dense layers
    'dense_units': [64, 32],
    'dense_dropout': [0.5, 0.3],
    
    # Training
    'epochs': 15,
    'batch_size': 32,
    'validation_split': 0.2,
    'early_stopping': True,
    'early_stopping_patience': 5,
    'reduce_lr': True,
    'reduce_lr_patience': 3,
    
    # Optimizer
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'loss': 'categorical_crossentropy'
}

# =============================================================================
# DASHBOARD CONFIGURATION
# =============================================================================

DASHBOARD_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True,
    'title': '🎭 Sentiment Analysis Dashboard',
    
    # Visualization
    'chart_style': 'seaborn',
    'color_scheme': {
        'positive': '#28a745',
        'negative': '#dc3545',
        'neutral': '#ffc107',
        'background': '#f8f9fa'
    },
    
    # Features
    'max_batch_size': 100,          # Maximum texts per batch
    'enable_confidence': True,      # Show confidence scores
    'show_wordcloud': False         # Enable wordcloud visualization
}

# =============================================================================
# OUTPUT & LOGGING CONFIGURATION
# =============================================================================

OUTPUT_CONFIG = {
    # Model saving
    'save_models': True,
    'model_format': 'pickle',       # 'pickle', 'joblib', 'h5' (for LSTM)
    
    # File names
    'preprocessor_file': 'preprocessor.pkl',
    'extractor_file': 'feature_extractor.pkl',
    'vectorizer_file': 'vectorizer.pkl',
    'lstm_model_file': 'lstm_model.h5',
    'lstm_tokenizer_file': 'lstm_tokenizer.pkl',
    
    # Logging
    'log_level': 'INFO',            # DEBUG, INFO, WARNING, ERROR
    'log_to_file': True,
    'log_file': 'logs/sentiment_analysis.log',
    
    # Results
    'save_predictions': True,
    'predictions_dir': 'predictions/'
}

# =============================================================================
# SENTIMENT MAPPING
# =============================================================================

# For datasets with numeric ratings (e.g., Amazon 1-5 stars)
SENTIMENT_MAPPING = {
    'rating_to_sentiment': {
        5: 'positive',
        4: 'positive',
        3: 'neutral',
        2: 'negative',
        1: 'negative'
    },
    'binary_mapping': {
        'positive': 1,
        'negative': 0,
        'neutral': 0.5
    }
}

# =============================================================================
# API CONFIGURATION (for future REST API deployment)
# =============================================================================

API_CONFIG = {
    'enabled': False,
    'host': '0.0.0.0',
    'port': 8000,
    'cors_enabled': True,
    'rate_limit': '100/hour',
    'api_key_required': False,
    'max_text_length': 1000,
    'response_format': 'json'
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_config(section=None):
    """
    Get configuration dictionary
    Usage: get_config('preprocessing') or get_config()
    """
    all_configs = {
        'data': DATA_CONFIG,
        'preprocessing': PREPROCESSING_CONFIG,
        'feature': FEATURE_CONFIG,
        'model': MODEL_CONFIG,
        'lstm': LSTM_CONFIG,
        'dashboard': DASHBOARD_CONFIG,
        'output': OUTPUT_CONFIG,
        'sentiment': SENTIMENT_MAPPING,
        'api': API_CONFIG,
        'paths': {
            'base': str(BASE_DIR),
            'data': str(DATA_DIR),
            'models': str(MODELS_DIR),
            'src': str(SRC_DIR),
            'app': str(APP_DIR)
        }
    }
    
    if section:
        return all_configs.get(section, {})
    return all_configs

def update_config(section, key, value):
    """Update a specific configuration value"""
    configs = {
        'data': DATA_CONFIG,
        'preprocessing': PREPROCESSING_CONFIG,
        'feature': FEATURE_CONFIG,
        'model': MODEL_CONFIG,
        'lstm': LSTM_CONFIG,
        'dashboard': DASHBOARD_CONFIG,
        'output': OUTPUT_CONFIG
    }
    
    if section in configs and key in configs[section]:
        configs[section][key] = value
        return True
    return False

def print_config():
    """Print current configuration summary"""
    print("="*60)
    print("📋 CURRENT CONFIGURATION")
    print("="*60)
    print(f"Dataset: {DATA_CONFIG['dataset_type']}")
    print(f"Sample Size: {DATA_CONFIG['sample_size']}")
    print(f"Feature Method: {FEATURE_CONFIG['primary_method']}")
    print(f"Max Features: {FEATURE_CONFIG['tfidf']['max_features']}")
    print(f"LSTM Enabled: {LSTM_CONFIG['enabled']}")
    print(f"Dashboard Port: {DASHBOARD_CONFIG['port']}")
    print("="*60)

# Run this to see current config
if __name__ == "__main__":
    print_config()