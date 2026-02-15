# 🎭 Sentiment Analysis AI System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-yellow.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

&gt; **Complete AI system for analyzing sentiment in Twitter posts and product reviews**
&gt; 
&gt; Deployed models: Naïve Bayes | Logistic Regression | LSTM Deep Learning

![Dashboard Preview](outputs/sentiment_dashboard.png)

## 🌟 Features

- ✅ **Text Preprocessing**: Cleaning, tokenization, stopword removal, slang handling
- ✅ **Multiple Models**: Naïve Bayes, Logistic Regression, LSTM Neural Networks
- ✅ **Feature Extraction**: TF-IDF, Word Embeddings
- ✅ **Interactive Dashboard**: Real-time sentiment visualization (% Pos/Neg/Neu)
- ✅ **REST API**: Flask-based web service for predictions
- ✅ **Real-World Ready**: Brand monitoring, customer feedback, political analysis

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/sentiment-analysis.git
cd sentiment-analysis

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"