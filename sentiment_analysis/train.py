import os
import sys
import time
import json
import pickle
import argparse
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_recall_fscore_support)

# Import project modules
from config import get_config, MODELS_DIR, DATA_DIR
from src.data_loader import DataLoader
from src.preprocessor import TextPreprocessor, download_nltk_data
from src.feature_extractor import FeatureExtractor
from src.models import TraditionalModels, LSTMModel

class ModelTrainer:
    """
    Complete training pipeline
    Handles data loading, preprocessing, feature extraction, and model training
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.data_config = self.config['data']
        self.prep_config = self.config['preprocessing']
        self.feature_config = self.config['feature']
        self.model_config = self.config['model']
        self.lstm_config = self.config['lstm']
        
        # Results storage
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'models': {},
            'best_model': None,
            'training_time': {}
        }
        
        # Data storage
        self.df = None
        self.df_processed = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None
        self.extractor = None
        
        # Create models directory
        MODELS_DIR.mkdir(exist_ok=True)
        
        self.print_banner()
    
    def print_banner(self):
        """Print training banner"""
        print("""
╔══════════════════════════════════════════════════════════════════╗
║              🤖 SENTIMENT ANALYSIS MODEL TRAINER                 ║
║                     Training Pipeline v1.0                        ║
╚══════════════════════════════════════════════════════════════════╝
        """)
    
    def log(self, message, level="INFO"):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        icons = {
            "INFO": "ℹ️ ",
            "SUCCESS": "✅",
            "WARNING": "⚠️ ",
            "ERROR": "❌",
            "STEP": "👉",
            "MODEL": "🎓",
            "DATA": "📊",
            "SAVE": "💾"
        }
        icon = icons.get(level, "•")
        print(f"[{timestamp}] {icon} {message}")
    
    def step(self, number, title):
        """Print step header"""
        print(f"\n{'='*70}")
        print(f"STEP {number}: {title}")
        print(f"{'='*70}")
    
    def step_1_load_data(self):
        """Step 1: Load and validate data"""
        self.step(1, "DATA LOADING")
        
        # Download NLTK data first
        self.log("Downloading NLTK resources...", "INFO")
        download_nltk_data()
        
        # Initialize data loader
        loader = DataLoader()
        dataset_type = self.data_config['dataset_type']
        
        self.log(f"Loading {dataset_type} dataset...", "DATA")
        
        # Load appropriate dataset
        if dataset_type == 'twitter':
            self.df = loader.load_twitter_data(self.data_config['twitter_data_path'])
        elif dataset_type == 'imdb':
            self.df = loader.load_imdb_data(self.data_config['imdb_data_path'])
        elif dataset_type == 'amazon':
            self.df = loader.load_amazon_data(self.data_config['amazon_data_path'])
        else:
            # Custom or default
            self.df = loader.load_twitter_data()  # Default to Twitter sample
        
        # Validate data
        if self.df is None or len(self.df) == 0:
            raise ValueError("No data loaded!")
        
        self.log(f"Loaded {len(self.df)} samples", "SUCCESS")
        self.log(f"Columns: {list(self.df.columns)}", "INFO")
        
        # Show distribution
        if 'sentiment' in self.df.columns:
            dist = self.df['sentiment'].value_counts()
            self.log("Class distribution:", "DATA")
            for sentiment, count in dist.items():
                pct = (count / len(self.df)) * 100
                print(f"   {sentiment}: {count} ({pct:.1f}%)")
        
        return self.df
    
    def step_2_preprocess(self):
        """Step 2: Text preprocessing"""
        self.step(2, "TEXT PREPROCESSING")
        
        self.log("Initializing preprocessor...", "INFO")
        self.preprocessor = TextPreprocessor(
            remove_stopwords=self.prep_config['remove_stopwords'],
            lemmatize=self.prep_config['lemmatization'],
            stem=self.prep_config['stemming']
        )
        
        # Identify text column
        text_col = None
        for col in ['text', 'review', 'review_text', 'content']:
            if col in self.df.columns:
                text_col = col
                break
        
        if text_col is None:
            raise ValueError("No text column found in dataframe")
        
        self.log(f"Processing text column: '{text_col}'", "INFO")
        
        # Preprocess
        self.df_processed = self.preprocessor.preprocess_dataframe(
            self.df, text_column=text_col
        )
        
        # Remove empty texts
        self.df_processed = self.df_processed[
            self.df_processed['processed_text'].str.len() > 0
        ]
        
        self.log(f"Preprocessing complete: {len(self.df_processed)} valid samples", "SUCCESS")
        
        # Show examples
        self.log("Preprocessing examples:", "DATA")
        for i in range(min(3, len(self.df_processed))):
            orig = self.df[text_col].iloc[i]
            proc = self.df_processed['processed_text'].iloc[i]
            print(f"\n   Original:  {orig[:80]}...")
            print(f"   Processed: {proc[:80]}...")
        
        # Statistics
        stats = self.preprocessor.get_word_stats(self.df_processed)
        self.log("Corpus statistics:", "DATA")
        self.log(f"  Total words: {stats['total_words']:,}", "INFO")
        self.log(f"  Unique words: {stats['unique_words']:,}", "INFO")
        self.log(f"  Avg words/text: {stats['avg_words_per_text']:.1f}", "INFO")
        
        return self.df_processed
    
    def step_3_feature_extraction(self):
        """Step 3: Extract features"""
        self.step(3, "FEATURE EXTRACTION")
        
        method = self.feature_config['primary_method']
        self.log(f"Using method: {method.upper()}", "INFO")
        
        self.extractor = FeatureExtractor(
            method=method,
            max_features=self.feature_config['tfidf']['max_features']
        )
        
        # Fit and transform
        X = self.extractor.fit_transform(self.df_processed['processed_text'])
        y = self.df_processed['sentiment'].values
        
        self.log(f"Feature matrix shape: {X.shape}", "SUCCESS")
        self.log(f"Features: {X.shape[1]:,}", "INFO")
        
        # Split data
        test_size = self.data_config['test_size']
        random_state = self.data_config['random_state']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.log(f"Train set: {len(self.y_train)} samples", "INFO")
        self.log(f"Test set: {len(self.y_test)} samples", "INFO")
        
        # Save feature extractor
        extractor_path = MODELS_DIR / self.config['output']['extractor_file']
        self.extractor.save(str(extractor_path))
        self.log(f"Feature extractor saved", "SAVE")
        
        return X, y
    
    def step_4_train_traditional_models(self):
        """Step 4: Train traditional ML models"""
        self.step(4, "TRADITIONAL ML MODELS")
        
        models = TraditionalModels()
        
        # Train Naïve Bayes
        if self.model_config['train_naive_bayes']:
            start_time = time.time()
            self.log("Training Naïve Bayes...", "MODEL")
            models.train_naive_bayes(
                self.X_train, self.y_train, 
                self.X_test, self.y_test
            )
            self.results['training_time']['naive_bayes'] = time.time() - start_time
        
        # Train Logistic Regression
        if self.model_config['train_logistic_regression']:
            start_time = time.time()
            self.log("Training Logistic Regression...", "MODEL")
            models.train_logistic_regression(
                self.X_train, self.y_train,
                self.X_test, self.y_test
            )
            self.results['training_time']['logistic_regression'] = time.time() - start_time
        
        # Train SVM
        if self.model_config['train_svm']:
            start_time = time.time()
            self.log("Training SVM...", "MODEL")
            models.train_svm(
                self.X_train, self.y_train,
                self.X_test, self.y_test
            )
            self.results['training_time']['svm'] = time.time() - start_time
        
        # Train Random Forest
        if self.model_config['train_random_forest']:
            start_time = time.time()
            self.log("Training Random Forest...", "MODEL")
            models.train_random_forest(
                self.X_train, self.y_train,
                self.X_test, self.y_test
            )
            self.results['training_time']['random_forest'] = time.time() - start_time
        
        # Get best model
        best_name, best_model = models.get_best_model()
        self.results['best_model'] = best_name
        self.results['models']['traditional'] = models.results
        
        # Save all models
        for model_name, model in models.models.items():
            model_path = MODELS_DIR / f"{model_name}.pkl"
            models.save_model(model_name, str(model_path))
            self.log(f"Saved {model_name}", "SAVE")
        
        return models
    
    def step_5_train_lstm(self):
        """Step 5: Train LSTM deep learning model"""
        if not self.lstm_config['enabled']:
            self.log("LSTM training disabled in config", "WARNING")
            return None
        
        self.step(5, "LSTM DEEP LEARNING MODEL")
        
        start_time = time.time()
        
        # Initialize LSTM
        lstm = LSTMModel(
            max_words=self.lstm_config['max_words'],
            max_len=self.lstm_config['max_len'],
            embedding_dim=self.lstm_config['embedding_dim']
        )
        
        # Prepare sequences
        self.log("Preparing sequences...", "INFO")
        X_lstm, y_lstm = lstm.prepare_sequences(
            self.df_processed['processed_text'],
            self.df_processed['sentiment']
        )
        
        # Split
        X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
            X_lstm, y_lstm,
            test_size=self.data_config['test_size'],
            random_state=self.data_config['random_state']
        )
        
        self.log(f"Sequence shape: {X_train_l.shape}", "INFO")
        
        # Build model
        self.log("Building LSTM architecture...", "INFO")
        lstm.build_model(num_classes=y_lstm.shape[1])
        
        # Train
        self.log(f"Training for max {self.lstm_config['epochs']} epochs...", "MODEL")
        history = lstm.train(
            X_train_l, y_train_l,
            X_test_l, y_test_l,
            epochs=self.lstm_config['epochs'],
            batch_size=self.lstm_config['batch_size']
        )
        
        # Evaluate
        self.log("Evaluating LSTM...", "INFO")
        loss, accuracy = lstm.model.evaluate(X_test_l, y_test_l, verbose=0)
        self.log(f"LSTM Test Accuracy: {accuracy:.4f}", "SUCCESS")
        
        # Save
        model_path = MODELS_DIR / self.config['output']['lstm_model_file']
        tokenizer_path = MODELS_DIR / self.config['output']['lstm_tokenizer_file']
        lstm.save(str(model_path), str(tokenizer_path))
        self.log("LSTM model saved", "SAVE")
        
        self.results['training_time']['lstm'] = time.time() - start_time
        self.results['models']['lstm'] = {
            'accuracy': accuracy,
            'loss': loss
        }
        
        return lstm
    
    def step_6_save_artifacts(self):
        """Step 6: Save all training artifacts"""
        self.step(6, "SAVING ARTIFACTS")
        
        # Save preprocessor
        prep_path = MODELS_DIR / self.config['output']['preprocessor_file']
        with open(prep_path, 'wb') as f:
            pickle.dump(self.preprocessor, f)
        self.log("Preprocessor saved", "SAVE")
        
        # Save training report
        report_path = MODELS_DIR / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        self.log(f"Training report saved: {report_path}", "SAVE")
        
        # Save config copy
        config_path = MODELS_DIR / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
        self.log("Configuration saved", "SAVE")
    
    def print_final_report(self):
        """Print final training report"""
        print(f"\n{'='*70}")
        print("📊 TRAINING COMPLETE - FINAL REPORT")
        print(f"{'='*70}")
        
        print(f"\n🏆 Best Model: {self.results['best_model']}")
        
        print(f"\n⏱️  Training Times:")
        for model, duration in self.results['training_time'].items():
            print(f"   {model}: {duration:.2f}s")
        
        print(f"\n📁 Saved Models:")
        models_dir = MODELS_DIR
        for file in sorted(models_dir.glob("*")):
            size = file.stat().st_size / 1024  # KB
            print(f"   {file.name} ({size:.1f} KB)")
        
        print(f"\n{'='*70}")
        print("✅ All models trained and ready for prediction!")
        print(f"{'='*70}")
        print("\nNext steps:")
        print("   1. python predict.py -t 'Your text here'")
        print("   2. cd app && python app.py  (Launch dashboard)")
        print(f"{'='*70}\n")
    
    def run_full_pipeline(self):
        """Execute complete training pipeline"""
        try:
            # Step 1: Load data
            self.step_1_load_data()
            
            # Step 2: Preprocess
            self.step_2_preprocess()
            
            # Step 3: Feature extraction
            self.step_3_feature_extraction()
            
            # Step 4: Train traditional models
            self.step_4_train_traditional_models()
            
            # Step 5: Train LSTM
            self.step_5_train_lstm()
            
            # Step 6: Save artifacts
            self.step_6_save_artifacts()
            
            # Final report
            self.print_final_report()
            
            return True
            
        except Exception as e:
            self.log(f"Training failed: {str(e)}", "ERROR")
            import traceback
            traceback.print_exc()
            return False
    
    def quick_train(self, model_type='logistic_regression'):
        """Quick train single model for testing"""
        self.log("QUICK TRAIN MODE", "STEP")
        
        # Load and preprocess
        self.step_1_load_data()
        self.step_2_preprocess()
        self.step_3_feature_extraction()
        
        # Train single model
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import MultinomialNB
        
        if model_type == 'logistic_regression':
            model = LogisticRegression(max_iter=1000)
        else:
            model = MultinomialNB()
        
        self.log(f"Training {model_type}...", "MODEL")
        model.fit(self.X_train, self.y_train)
        
        # Evaluate
        score = model.score(self.X_test, self.y_test)
        self.log(f"Accuracy: {score:.4f}", "SUCCESS")
        
        # Save
        model_path = MODELS_DIR / f"{model_type}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save preprocessor and extractor
        prep_path = MODELS_DIR / "preprocessor.pkl"
        with open(prep_path, 'wb') as f:
            pickle.dump(self.preprocessor, f)
        
        extractor_path = MODELS_DIR / "feature_extractor.pkl"
        self.extractor.save(str(extractor_path))
        
        self.log("Quick training complete!", "SUCCESS")


def main():
    """Command line interface for training"""
    parser = argparse.ArgumentParser(
        description='Train Sentiment Analysis Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full training pipeline (recommended)
  python train.py
  
  # Quick train (single model, for testing)
  python train.py --quick
  
  # Train only traditional models (no LSTM)
  python train.py --no-lstm
  
  # Train only LSTM
  python train.py --lstm-only
  
  # Resume from saved state
  python train.py --resume
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Quick train with single model')
    parser.add_argument('--no-lstm', action='store_true',
                       help='Skip LSTM training')
    parser.add_argument('--lstm-only', action='store_true',
                       help='Train only LSTM model')
    parser.add_argument('--model', default='logistic_regression',
                       choices=['naive_bayes', 'logistic_regression', 'svm', 'lstm'],
                       help='Model for quick train')
    parser.add_argument('--config', help='Path to custom config file')
    
    args = parser.parse_args()
    
    # Load config
    config = get_config()
    if args.no_lstm:
        config['lstm']['enabled'] = False
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    if args.quick:
        trainer.quick_train(args.model)
    elif args.lstm_only:
        # Load existing preprocessed data if available
        trainer.step_1_load_data()
        trainer.step_2_preprocess()
        trainer.step_3_feature_extraction()
        trainer.step_5_train_lstm()
    else:
        # Full pipeline
        success = trainer.run_full_pipeline()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()