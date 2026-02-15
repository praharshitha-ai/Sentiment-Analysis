import re
import string
import unicodedata
from collections import Counter
from typing import List, Dict, Union, Optional
import warnings

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, TweetTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download required NLTK data
def download_nltk_data():
    """Download all required NLTK data packages"""
    required_packages = [
        'punkt', 'stopwords', 'wordnet', 'vader_lexicon',
        'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words'
    ]
    
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
        except LookupError:
            print(f"📥 Downloading {package}...")
            nltk.download(package, quiet=True)
    
    print("✅ All NLTK data ready")

class TextPreprocessor:
    """
    Advanced text preprocessor with multiple cleaning options
    """
    
    def __init__(self, 
                 lowercase=True,
                 remove_urls=True,
                 remove_mentions=True,
                 remove_hashtags=False,
                 remove_emojis=False,
                 remove_punctuation=True,
                 remove_numbers=False,
                 remove_stopwords=True,
                 custom_stopwords=None,
                 lemmatize=True,
                 stem=False,
                 stemmer='wordnet',  # 'porter', 'snowball', 'wordnet'
                 min_word_length=2,
                 max_word_length=20,
                 language='english',
                 preserve_case_for_emojis=True):
        
        self.config = {
            'lowercase': lowercase,
            'remove_urls': remove_urls,
            'remove_mentions': remove_mentions,
            'remove_hashtags': remove_hashtags,
            'remove_emojis': remove_emojis,
            'remove_punctuation': remove_punctuation,
            'remove_numbers': remove_numbers,
            'remove_stopwords': remove_stopwords,
            'lemmatize': lemmatize,
            'stem': stem,
            'min_word_length': min_word_length,
            'max_word_length': max_word_length,
            'language': language
        }
        
        # Initialize tools
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = self._get_stemmer(stemmer, language)
        self.tweet_tokenizer = TweetTokenizer(preserve_case=not lowercase, 
                                              reduce_len=True, 
                                              strip_handles=remove_mentions)
        
        # Stopwords
        self.stop_words = set(stopwords.words(language))
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)
        
        # Add domain-specific stopwords
        self.domain_stopwords = {
            'rt', 'via', 'amp', 'http', 'https', 'co', 'www',
            'com', 'org', 'net', 'io', 'app'
        }
        self.stop_words.update(self.domain_stopwords)
        
        # Sentiment analyzer (for augmentation)
        self.sia = SentimentIntensityAnalyzer()
        
        # Emoji pattern
        self.emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        
        # URL pattern
        self.url_pattern = re.compile(
            r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
            r'|www\.(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        )
        
        # Mention pattern
        self.mention_pattern = re.compile(r'@\w+')
        
        # Hashtag pattern
        self.hashtag_pattern = re.compile(r'#(\w+)')
        
        # Number pattern
        self.number_pattern = re.compile(r'\d+')
        
        print(f"🔧 TextPreprocessor initialized")
        print(f"   Language: {language}")
        print(f"   Lemmatization: {lemmatize}")
        print(f"   Stemming: {stem}")
        print(f"   Stopwords: {len(self.stop_words)} words")
    
    def _get_stemmer(self, stemmer_type, language):
        """Get appropriate stemmer"""
        if stemmer_type == 'porter':
            return PorterStemmer()
        elif stemmer_type == 'snowball':
            return SnowballStemmer(language)
        return None
    
    def clean_text(self, text: str) -> str:
        """
        Apply all cleaning steps to text
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Remove URLs
        if self.config['remove_urls']:
            text = self.url_pattern.sub('', text)
        
        # Remove mentions
        if self.config['remove_mentions']:
            text = self.mention_pattern.sub('', text)
        
        # Handle hashtags
        if self.config['remove_hashtags']:
            text = self.hashtag_pattern.sub(r'\1', text)  # Keep word, remove #
        else:
            text = text.replace('#', '')  # Just remove # symbol
        
        # Remove emojis
        if self.config['remove_emojis']:
            text = self.emoji_pattern.sub('', text)
        
        # Remove numbers
       