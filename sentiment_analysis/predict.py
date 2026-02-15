from src.predict import SentimentPredictor

predictor = SentimentPredictor.load('models/best_model.pkl')
result = predictor.predict("This product is amazing!")
print(result)
# {'sentiment': 'Positive', 'confidence': 0.95, ...}#!/usr/bin/env python3
"""
PREDICTION MODULE
Make sentiment predictions on new text data
Can be used standalone or imported into other modules
"""

import sys
import os
import pickle
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import get_config, MODELS_DIR
from src.preprocessor import TextPreprocessor
from src.feature_extractor import FeatureExtractor
from src.models import LSTMModel

class SentimentPredictor:
    """
    Main prediction class
    Loads trained models and makes predictions on new text
    """
    
    def __init__(self, model_type='best'):
        """
        Initialize predictor
        Args:
            model_type: 'best', 'naive_bayes', 'logistic_regression', 'svm', 'lstm'
        """
        self.config = get_config()
        self.model_type = model_type
        self.preprocessor = None
        self.extractor = None
        self.model = None
        self.lstm_model = None
        self.label_mapping = None
        
        self.models_dir = Path(MODELS_DIR)
        self._load_models()
    
    def _load_models(self):
        """Load all necessary models"""
        print("📂 Loading models...")
        
        # Load preprocessor
        preprocessor_path = self.models_dir / self.config['output']['preprocessor_file']
        if preprocessor_path.exists():
            with open(preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
            print("✅ Preprocessor loaded")
        else:
            print("⚠️  Preprocessor not found, using default")
            self.preprocessor = TextPreprocessor()
        
        # Load feature extractor (for traditional models)
        if self.model_type != 'lstm':
            extractor_path = self.models_dir / self.config['output']['extractor_file']
            if extractor_path.exists():
                self.extractor = FeatureExtractor(method='tfidf')
                self.extractor.load(str(extractor_path))
                print("✅ Feature extractor loaded")
        
        # Load prediction model
        if self.model_type == 'lstm':
            self._load_lstm()
        else:
            self._load_traditional_model()
    
    def _load_traditional_model(self):
        """Load traditional ML model"""
        if self.model_type == 'best':
            # Find the best model (highest F1 score)
            model_files = list(self.models_dir.glob('*.pkl'))
            model_files = [f for f in model_files 
                          if 'extractor' not in f.name and 'tokenizer' not in f.name 
                          and 'preprocessor' not in f.name]
            
            if not model_files:
                raise FileNotFoundError("No trained models found in models/ directory")
            
            # Use the first available model as best
            model_path = model_files[0]
            self.model_type = model_path.stem
        else:
            model_path = self.models_dir / f"{self.model_type}.pkl"
        
        if model_path.exists():
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✅ Model loaded: {self.model_type}")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
    
    def _load_lstm(self):
        """Load LSTM model"""
        model_path = self.models_dir / self.config['output']['lstm_model_file']
        tokenizer_path = self.models_dir / self.config['output']['lstm_tokenizer_file']
        
        if not model_path.exists():
            raise FileNotFoundError(f"LSTM model not found: {model_path}")
        
        self.lstm_model = LSTMModel(
            max_words=self.config['lstm']['max_words'],
            max_len=self.config['lstm']['max_len']
        )
        self.lstm_model.load(str(model_path), str(tokenizer_path))
        print("✅ LSTM model loaded")
    
    def predict(self, text, return_confidence=True):
        """
        Predict sentiment for a single text
        Args:
            text: Input text string
            return_confidence: Whether to return confidence score
        Returns:
            dict with sentiment, confidence, and processed text
        """
        if isinstance(text, list):
            return self.predict_batch(text, return_confidence)
        
        # Preprocess
        processed_text = self.preprocessor.preprocess(text)
        
        if not processed_text.strip():
            return {
                'text': text,
                'processed_text': processed_text,
                'sentiment': 'neutral',
                'confidence': 0.0,
                'model': self.model_type
            }
        
        # Predict based on model type
        if self.model_type == 'lstm':
            sentiment, confidence = self.lstm_model.predict([processed_text])
            sentiment = sentiment[0]
            confidence = confidence[0]
        else:
            # Traditional model
            features = self.extractor.transform([processed_text])
            sentiment = self.model.predict(features)[0]
            
            # Get confidence if available
            if return_confidence and hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features)[0]
                confidence = np.max(proba)
            else:
                confidence = 0.8  # Default confidence
        
        return {
            'text': text,
            'processed_text': processed_text,
            'sentiment': sentiment,
            'confidence': float(confidence),
            'model': self.model_type
        }
    
    def predict_batch(self, texts, return_confidence=True):
        """
        Predict sentiment for multiple texts
        Args:
            texts: List of text strings
            return_confidence: Whether to return confidence scores
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for text in texts:
            result = self.predict(text, return_confidence)
            results.append(result)
        
        return results
    
    def predict_dataframe(self, df, text_column='text'):
        """
        Predict sentiment for entire dataframe
        Args:
            df: pandas DataFrame
            text_column: Name of column containing text
        Returns:
            DataFrame with added 'sentiment' and 'confidence' columns
        """
        texts = df[text_column].tolist()
        predictions = self.predict_batch(texts)
        
        df['predicted_sentiment'] = [p['sentiment'] for p in predictions]
        df['confidence'] = [p['confidence'] for p in predictions]
        df['processed_text'] = [p['processed_text'] for p in predictions]
        
        return df
    
    def get_sentiment_distribution(self, texts):
        """
        Get sentiment distribution statistics
        Args:
            texts: List of text strings
        Returns:
            Dictionary with counts and percentages
        """
        predictions = self.predict_batch(texts)
        
        sentiments = [p['sentiment'] for p in predictions]
        counts = {
            'positive': sentiments.count('positive'),
            'negative': sentiments.count('negative'),
            'neutral': sentiments.count('neutral')
        }
        
        total = len(sentiments)
        percentages = {
            'positive': (counts['positive'] / total) * 100 if total > 0 else 0,
            'negative': (counts['negative'] / total) * 100 if total > 0 else 0,
            'neutral': (counts['neutral'] / total) * 100 if total > 0 else 0
        }
        
        return {
            'counts': counts,
            'percentages': percentages,
            'total': total
        }
    
    def analyze_file(self, filepath, text_column='text', output_path=None):
        """
        Analyze a CSV or Excel file
        Args:
            filepath: Path to input file
            text_column: Column name containing text
            output_path: Optional path to save results
        Returns:
            DataFrame with predictions
        """
        # Load file
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")
        
        print(f"📊 Loaded {len(df)} rows from {filepath}")
        
        # Predict
        df_result = self.predict_dataframe(df, text_column)
        
        # Save if output path provided
        if output_path:
            if output_path.endswith('.csv'):
                df_result.to_csv(output_path, index=False)
            elif output_path.endswith('.xlsx'):
                df_result.to_excel(output_path, index=False)
            print(f"💾 Results saved to {output_path}")
        
        # Print summary
        dist = self.get_sentiment_distribution(df[text_column].tolist())
        print("\n📈 Sentiment Distribution:")
        print(f"   Positive: {dist['counts']['positive']} ({dist['percentages']['positive']:.1f}%)")
        print(f"   Neutral:  {dist['counts']['neutral']} ({dist['percentages']['neutral']:.1f}%)")
        print(f"   Negative: {dist['counts']['negative']} ({dist['percentages']['negative']:.1f}%)")
        
        return df_result


def main():
    """Command line interface for prediction"""
    parser = argparse.ArgumentParser(
        description='Sentiment Analysis Prediction Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict single text
  python predict.py -t "I love this product!"
  
  # Predict multiple texts
  python predict.py -t "Great service" "Terrible experience" "It's okay"
  
  # Analyze CSV file
  python predict.py -f data/reviews.csv -c review_text -o results.csv
  
  # Use specific model
  python predict.py -m lstm -t "Amazing quality!"
        """
    )
    
    parser.add_argument('-t', '--text', nargs='+', help='Text(s) to analyze')
    parser.add_argument('-f', '--file', help='Input CSV/Excel file path')
    parser.add_argument('-c', '--column', default='text', help='Text column name (default: text)')
    parser.add_argument('-o', '--output', help='Output file path (CSV or Excel)')
    parser.add_argument('-m', '--model', default='best', 
                       choices=['best', 'naive_bayes', 'logistic_regression', 'svm', 'lstm'],
                       help='Model to use for prediction (default: best)')
    parser.add_argument('--no-confidence', action='store_true', help='Hide confidence scores')
    parser.add_argument('--format', choices=['text', 'json'], default='text', help='Output format')
    
    args = parser.parse_args()
    
    # Initialize predictor
    try:
        predictor = SentimentPredictor(model_type=args.model)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please train models first by running: python main.py")
        return
    
    # Single or batch text prediction
    if args.text:
        if len(args.text) == 1:
            # Single prediction
            result = predictor.predict(args.text[0], not args.no_confidence)
            
            if args.format == 'json':
                print(json.dumps(result, indent=2))
            else:
                print(f"\n📝 Text: {result['text']}")
                print(f"🔧 Processed: {result['processed_text']}")
                print(f"😊 Sentiment: {result['sentiment'].upper()}")
                if not args.no_confidence:
                    print(f"📊 Confidence: {result['confidence']*100:.1f}%")
                print(f"🤖 Model: {result['model']}")
        else:
            # Batch prediction
            results = predictor.predict_batch(args.text, not args.no_confidence)
            
            if args.format == 'json':
                print(json.dumps(results, indent=2))
            else:
                print(f"\n📊 Batch Prediction Results ({len(results)} texts):")
                print("-" * 60)
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result['text'][:50]}...")
                    print(f"   Sentiment: {result['sentiment'].upper()}")
                    if not args.no_confidence:
                        print(f"   Confidence: {result['confidence']*100:.1f}%")
                    print()
    
    # File prediction
    elif args.file:
        try:
            df_result = predictor.analyze_file(args.file, args.column, args.output)
            
            if not args.output:
                # Show first few results
                print("\n📋 Sample Results:")
                print(df_result[[args.column, 'predicted_sentiment', 'confidence']].head(10).to_string())
                
        except Exception as e:
            print(f"❌ Error processing file: {e}")
    
    else:
        parser.print_help()
        print("\n💡 Tip: Use -t 'your text here' to analyze text")


# Simple API for importing into other projects
def quick_predict(text, model_type='best'):
    """
    Quick prediction function for external use
    Example:
        from predict import quick_predict
        result = quick_predict("I love this!")
    """
    predictor = SentimentPredictor(model_type=model_type)
    return predictor.predict(text)

def batch_predict(texts, model_type='best'):
    """
    Batch prediction for external use
    Example:
        from predict import batch_predict
        results = batch_predict(["text1", "text2", "text3"])
    """
    predictor = SentimentPredictor(model_type=model_type)
    return predictor.predict_batch(texts)


if __name__ == "__main__":
    main()