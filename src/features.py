import numpy as np
import pandas as pd
from collections import Counter
import re
import string

# Text features
from sklearn.feature_extraction.text import (
    TfidfVectorizer, CountVectorizer, 
    HashingVectorizer, TfidfTransformer
)
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler, normalize

# Deep learning features
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Sentiment lexicons
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

# Word embeddings
try:
    from gensim.models import Word2Vec, FastText
    from gensim.models.keyedvectors import KeyedVectors
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

# Transformer embeddings (optional)
try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class TextFeatureEngineer:
    """
    Advanced text feature engineering
    Creates multiple types of features for sentiment analysis
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.vectorizers = {}
        self.scalers = {}
        self.fitted = False
        
        # Initialize sentiment analyzers
        self.vader = None
        if VADER_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()
    
    # ==================== STATISTICAL FEATURES ====================
    
    def extract_statistical_features(self, texts):
        """
        Extract statistical features from text
        Returns DataFrame with features
        """
        features = []
        
        for text in texts:
            text_str = str(text)
            words = text_str.split()
            chars = list(text_str)
            
            # Basic counts
            feat = {
                'char_count': len(chars),
                'word_count': len(words),
                'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
                'sentence_count': text_str.count('.') + text_str.count('!') + text_str.count('?'),
                
                # Character features
                'uppercase_ratio': sum(1 for c in chars if c.isupper()) / len(chars) if chars else 0,
                'digit_ratio': sum(1 for c in chars if c.isdigit()) / len(chars) if chars else 0,
                'punctuation_ratio': sum(1 for c in chars if c in string.punctuation) / len(chars) if chars else 0,
                
                # Special characters
                'exclamation_count': text_str.count('!'),
                'question_count': text_str.count('?'),
                'ellipsis_count': text_str.count('...'),
                
                # Word features
                'unique_word_ratio': len(set(words)) / len(words) if words else 0,
                'avg_syllables_per_word': self._count_syllables_in_text(text_str),
                
                # Sentiment-specific
                'positive_words': self._count_sentiment_words(text_str, 'positive'),
                'negative_words': self._count_sentiment_words(text_str, 'negative'),
                'intensifiers': self._count_intensifiers(text_str),
            }
            features.append(feat)
        
        return pd.DataFrame(features)
    
    def _count_syllables(self, word):
        """Count syllables in a word"""
        word = word.lower()
        vowels = "aeiouy"
        syllables = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllables += 1
            prev_was_vowel = is_vowel
        
        # Handle silent e
        if word.endswith('e') and syllables > 1:
            syllables -= 1
        
        return max(1, syllables)
    
    def _count_syllables_in_text(self, text):
        """Average syllables per word in text"""
        words = text.split()
        if not words:
            return 0
        return np.mean([self._count_syllables(w) for w in words])
    
    def _count_sentiment_words(self, text, sentiment_type):
        """Count positive/negative words"""
        positive_words = {'good', 'great', 'excellent', 'amazing', 'love', 'best', 
                         'fantastic', 'wonderful', 'perfect', 'awesome', 'happy'}
        negative_words = {'bad', 'terrible', 'awful', 'hate', 'worst', 'horrible',
                         'disappointing', 'poor', 'sad', 'angry', 'frustrated'}
        
        words = set(text.lower().split())
        
        if sentiment_type == 'positive':
            return len(words & positive_words)
        else:
            return len(words & negative_words)
    
    def _count_intensifiers(self, text):
        """Count intensifier words"""
        intensifiers = {'very', 'really', 'extremely', 'incredibly', 'absolutely',
                       'completely', 'totally', 'quite', 'rather', 'pretty'}
        words = text.lower().split()
        return sum(1 for w in words if w in intensifiers)
    
    # ==================== LEXICON-BASED FEATURES ====================
    
    def extract_sentiment_scores(self, texts):
        """
        Extract sentiment scores using VADER and TextBlob
        """
        scores = []
        
        for text in texts:
            text_str = str(text)
            feat = {}
            
            # VADER scores
            if self.vader:
                vader_scores = self.vader.polarity_scores(text_str)
                feat['vader_compound'] = vader_scores['compound']
                feat['vader_positive'] = vader_scores['pos']
                feat['vader_negative'] = vader_scores['neg']
                feat['vader_neutral'] = vader_scores['neu']
            
            # TextBlob scores
            if TEXTBLOB_AVAILABLE:
                blob = TextBlob(text_str)
                feat['textblob_polarity'] = blob.sentiment.polarity
                feat['textblob_subjectivity'] = blob.sentiment.subjectivity
            
            scores.append(feat)
        
        return pd.DataFrame(scores)
    
    # ==================== N-GRAM FEATURES ====================
    
    def extract_ngram_features(self, texts, ngram_range=(1, 3), max_features=5000):
        """
        Extract character and word n-gram features
        """
        # Character n-grams (good for catching misspellings/slang)
        char_vectorizer = CountVectorizer(
            analyzer='char',
            ngram_range=(2, 5),  # Character n-grams
            max_features=max_features // 2,
            min_df=2
        )
        
        # Word n-grams
        word_vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features // 2,
            min_df=2,
            max_df=0.95
        )
        
        # Fit and transform
        char_features = char_vectorizer.fit_transform(texts)
        word_features = word_vectorizer.fit_transform(texts)
        
        # Store vectorizers
        self.vectorizers['char_ngram'] = char_vectorizer
        self.vectorizers['word_ngram'] = word_vectorizer
        
        return char_features, word_features
    
    # ==================== TOPIC FEATURES ====================
    
    def extract_topic_features(self, texts, n_topics=10, method='lda'):
        """
        Extract topic features using LDA or SVD
        """
        # Create TF-IDF first
        tfidf = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.95)
        tfidf_matrix = tfidf.fit_transform(texts)
        
        if method == 'lda':
            # Latent Dirichlet Allocation
            model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10
            )
        else:
            # Truncated SVD (LSA)
            model = TruncatedSVD(n_components=n_topics, random_state=42)
        
        topic_features = model.fit_transform(tfidf_matrix)
        
        self.vectorizers['tfidf_topic'] = tfidf
        self.vectorizers['topic_model'] = model
        
        return topic_features
    
    # ==================== EMBEDDING FEATURES ====================
    
    def extract_word2vec_features(self, texts, vector_size=100, window=5, min_count=1):
        """
        Extract Word2Vec features (average of word vectors)
        """
        if not GENSIM_AVAILABLE:
            raise ImportError("Gensim not installed. Install with: pip install gensim")
        
        # Tokenize
        tokenized_texts = [text.split() for text in texts]
        
        # Train Word2Vec
        model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
            sg=1,  # Skip-gram
            epochs=10
        )
        
        # Create document vectors by averaging word vectors
        doc_vectors = []
        for tokens in tokenized_texts:
            vectors = []
            for token in tokens:
                if token in model.wv:
                    vectors.append(model.wv[token])
            
            if vectors:
                doc_vectors.append(np.mean(vectors, axis=0))
            else:
                doc_vectors.append(np.zeros(vector_size))
        
        self.vectorizers['word2vec'] = model
        
        return np.array(doc_vectors)
    
    def extract_fasttext_features(self, texts, vector_size=100):
        """
        Extract FastText features (subword information)
        """
        if not GENSIM_AVAILABLE:
            raise ImportError("Gensim not installed")
        
        tokenized_texts = [text.split() for text in texts]
        
        model = FastText(
            sentences=tokenized_texts,
            vector_size=vector_size,
            window=3,
            min_count=1,
            workers=4,
            epochs=10
        )
        
        # Average vectors
        doc_vectors = []
        for tokens in tokenized_texts:
            vectors = [model.wv[token] for token in tokens if token in model.wv]
            if vectors:
                doc_vectors.append(np.mean(vectors, axis=0))
            else:
                doc_vectors.append(np.zeros(vector_size))
        
        self.vectorizers['fasttext'] = model
        
        return np.array(doc_vectors)
    
    def extract_transformer_embeddings(self, texts, model_name='all-MiniLM-L6-v2'):
        """
        Extract transformer-based sentence embeddings
        Requires: pip install sentence-transformers
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Install with: pip install sentence-transformers")
        
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, show_progress_bar=True)
        
        return embeddings
    
    # ==================== DEEP LEARNING FEATURES ====================
    
    def extract_sequence_features(self, texts, max_words=10000, max_len=100):
        """
        Prepare sequences for neural networks
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not installed")
        
        tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
        tokenizer.fit_on_texts(texts)
        
        sequences = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
        
        self.vectorizers['tokenizer'] = tokenizer
        
        return padded, tokenizer
    
    # ==================== COMBINED PIPELINE ====================
    
    def fit_transform(self, texts, feature_types='all'):
        """
        Extract all specified features and combine them
        
        feature_types: 'all', 'statistical', 'lexicon', 'ngram', 'topic', 'embedding'
        """
        all_features = []
        feature_names = []
        
        # 1. Statistical features
        if feature_types in ['all', 'statistical']:
            print("Extracting statistical features...")
            stat_features = self.extract_statistical_features(texts)
            all_features.append(stat_features.values)
            feature_names.extend(stat_features.columns.tolist())
            print(f"  Added {stat_features.shape[1]} statistical features")
        
        # 2. Lexicon features
        if feature_types in ['all', 'lexicon']:
            print("Extracting lexicon features...")
            lex_features = self.extract_sentiment_scores(texts)
            all_features.append(lex_features.values)
            feature_names.extend(lex_features.columns.tolist())
            print(f"  Added {lex_features.shape[1]} lexicon features")
        
        # 3. TF-IDF features
        if feature_types in ['all', 'ngram']:
            print("Extracting TF-IDF features...")
            tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
            tfidf_features = tfidf.fit_transform(texts).toarray()
            all_features.append(tfidf_features)
            self.vectorizers['tfidf'] = tfidf
            print(f"  Added {tfidf_features.shape[1]} TF-IDF features")
        
        # 4. Word2Vec features
        if feature_types in ['all', 'embedding'] and GENSIM_AVAILABLE:
            print("Extracting Word2Vec features...")
            w2v_features = self.extract_word2vec_features(texts, vector_size=100)
            all_features.append(w2v_features)
            print(f"  Added {w2v_features.shape[1]} Word2Vec features")
        
        # Combine all features
        if all_features:
            combined = np.hstack(all_features)
            print(f"\n✅ Total features: {combined.shape[1]}")
            self.fitted = True
            return combined, feature_names
        else:
            return None, []
    
    def transform(self, texts):
        """Transform new texts using fitted vectorizers"""
        if not self.fitted:
            raise ValueError("Must call fit_transform first!")
        
        # Similar to fit_transform but using stored vectorizers
        # Implementation depends on which features were used
        pass


class FeatureSelector:
    """
    Feature selection and dimensionality reduction
    """
    
    def __init__(self, method='chi2', k=1000):
        self.method = method
        self.k = k
        self.selector = None
    
    def fit_transform(self, X, y):
        """Select top k features"""
        from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
        
        if self.method == 'chi2':
            self.selector = SelectKBest(chi2, k=self.k)
        elif self.method == 'mutual_info':
            self.selector = SelectKBest(mutual_info_classif, k=self.k)
        elif self.method == 'f_classif':
            self.selector = SelectKBest(f_classif, k=self.k)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return self.selector.fit_transform(X, y)
    
    def transform(self, X):
        return self.selector.transform(X)


# ==================== UTILITY FUNCTIONS ====================

def create_feature_pipeline(texts, labels, config=None):
    """Create complete feature pipeline"""
    engineer = TextFeatureEngineer(config)
    
    # Extract features
    features, names = engineer.fit_transform(texts, 'all')
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Select features
    selector = FeatureSelector(method='chi2', k=min(1000, features.shape[1]))
    features_selected = selector.fit_transform(features_scaled, labels)
    
    return {
        'features': features_selected,
        'feature_names': names,
        'engineer': engineer,
        'scaler': scaler,
        'selector': selector
    }


if __name__ == "__main__":
    # Test features module
    print("Testing Features Module...")
    
    sample_texts = [
        "I absolutely love this amazing product! Best ever!!! 😍",
        "Terrible experience. Waste of money. Very disappointed.",
        "It's okay, nothing special but does the job.",
        "Outstanding quality! Highly recommend to everyone!"
    ]
    
    engineer = TextFeatureEngineer()
    
    # Test statistical features
    stat_df = engineer.extract_statistical_features(sample_texts)
    print(f"\n✅ Statistical features: {stat_df.shape[1]} features")
    print(stat_df.head())
    
    # Test lexicon features
    if VADER_AVAILABLE:
        lex_df = engineer.extract_sentiment_scores(sample_texts)
        print(f"\n✅ Lexicon features: {lex_df.shape[1]} features")
        print(lex_df.head())
    
    print("\n✅ Features module ready!")