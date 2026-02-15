import sys
import os

# Add parent directory to path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import json
import io
import base64
from datetime import datetime

# Import our modules
from src.data_loader import DataLoader
from src.preprocessor import TextPreprocessor, download_nltk_data
from src.feature_extractor import FeatureExtractor
from src.models import TraditionalModels, LSTMModel

# Visualization
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Wedge
import seaborn as sns

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Global model storage
model_store = {
    'preprocessor': None,
    'extractor': None,
    'traditional_models': None,
    'lstm_model': None,
    'best_model_name': None,
    'best_model': None
}

def create_pie_chart(sentiment_counts):
    """Create sentiment distribution pie chart"""
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#f8f9fa')
    
    colors = {'positive': '#28a745', 'negative': '#dc3545', 'neutral': '#ffc107'}
    labels = list(sentiment_counts.keys())
    sizes = list(sentiment_counts.values())
    chart_colors = [colors.get(l, '#6c757d') for l in labels]
    
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=chart_colors, autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 12, 'weight': 'bold'}
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
    
    ax.set_title('Sentiment Distribution', fontsize=16, fontweight='bold', pad=20)
    
    # Add center circle for donut effect
    centre_circle = plt.Circle((0, 0), 0.70, fc='#f8f9fa')
    ax.add_artist(centre_circle)
    
    # Add total count in center
    total = sum(sizes)
    ax.text(0, 0, f'Total\n{total}', ha='center', va='center', 
            fontsize=14, fontweight='bold')
    
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return base64.b64encode(buf.read()).decode('utf-8')

def create_bar_chart(sentiment_counts):
    """Create sentiment bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#f8f9fa')
    
    colors = {'positive': '#28a745', 'negative': '#dc3545', 'neutral': '#ffc107'}
    labels = list(sentiment_counts.keys())
    values = list(sentiment_counts.values())
    bar_colors = [colors.get(l, '#6c757d') for l in labels]
    
    bars = ax.bar(labels, values, color=bar_colors, edgecolor='white', linewidth=2)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Sentiment Analysis Results', fontsize=16, fontweight='bold', pad=20)
    ax.set_facecolor('#f8f9fa')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return base64.b64encode(buf.read()).decode('utf-8')

@app.route('/')
def dashboard():
    """Main dashboard page"""
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🎭 Sentiment Analysis Dashboard</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #333;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            header {
                text-align: center;
                color: white;
                padding: 40px 0;
            }
            header h1 {
                font-size: 3em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            }
            header p {
                font-size: 1.2em;
                opacity: 0.9;
            }
            .card {
                background: white;
                border-radius: 20px;
                padding: 30px;
                margin: 20px 0;
                box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            }
            .input-section {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }
            textarea {
                width: 100%;
                height: 150px;
                padding: 15px;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                font-size: 16px;
                resize: vertical;
                transition: border-color 0.3s;
            }
            textarea:focus {
                outline: none;
                border-color: #667eea;
            }
            button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 15px 40px;
                font-size: 18px;
                border-radius: 30px;
                cursor: pointer;
                transition: transform 0.3s, box-shadow 0.3s;
                font-weight: bold;
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 20px;
                margin: 20px 0;
            }
            .stat-box {
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                color: white;
                transition: transform 0.3s;
            }
            .stat-box:hover {
                transform: translateY(-5px);
            }
            .positive { background: linear-gradient(135deg, #11998e, #38ef7d); }
            .negative { background: linear-gradient(135deg, #eb3349, #f45c43); }
            .neutral { background: linear-gradient(135deg, #f7971e, #ffd200); }
            .stat-number {
                font-size: 3em;
                font-weight: bold;
                margin: 10px 0;
            }
            .stat-label {
                font-size: 1.2em;
                opacity: 0.9;
            }
            .chart-container {
                text-align: center;
                margin: 20px 0;
            }
            .chart-container img {
                max-width: 100%;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            .results-table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }
            .results-table th, .results-table td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #e0e0e0;
            }
            .results-table th {
                background: #f8f9fa;
                font-weight: bold;
            }
            .badge {
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 0.9em;
                font-weight: bold;
                color: white;
                text-transform: uppercase;
            }
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            @media (max-width: 768px) {
                .input-section { grid-template-columns: 1fr; }
                .stats-grid { grid-template-columns: 1fr; }
                header h1 { font-size: 2em; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🎭 Sentiment Analysis Dashboard</h1>
                <p>AI-Powered Text Analysis | Naïve Bayes | Logistic Regression | LSTM</p>
            </header>
            
            <div class="card">
                <h2 style="margin-bottom: 20px;">📝 Analyze Text</h2>
                <div class="input-section">
                    <div>
                        <textarea id="textInput" placeholder="Enter text to analyze (one per line for batch analysis)..."></textarea>
                        <div style="margin-top: 15px; text-align: center;">
                            <button onclick="analyzeText()">🔍 Analyze Sentiment</button>
                        </div>
                    </div>
                    <div>
                        <h3 style="margin-bottom: 15px;">📊 Quick Stats</h3>
                        <div class="stats-grid" id="quickStats">
                            <div class="stat-box positive">
                                <div class="stat-label">Positive</div>
                                <div class="stat-number" id="posCount">0</div>
                            </div>
                            <div class="stat-box neutral">
                                <div class="stat-label">Neutral</div>
                                <div class="stat-number" id="neuCount">0</div>
                            </div>
                            <div class="stat-box negative">
                                <div class="stat-label">Negative</div>
                                <div class="stat-number" id="negCount">0</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p style="margin-top: 10px;">Analyzing sentiment...</p>
                </div>
                
                <div id="results" style="display: none;">
                    <h3 style="margin: 20px 0;">📈 Visualization</h3>
                    <div class="chart-container">
                        <img id="pieChart" alt="Sentiment Distribution">
                    </div>
                    <div class="chart-container">
                        <img id="barChart" alt="Sentiment Bar Chart">
                    </div>
                    
                    <h3 style="margin: 20px 0;">📋 Detailed Results</h3>
                    <table class="results-table">
                        <thead>
                            <tr>
                                <th>Text</th>
                                <th>Sentiment</th>
                                <th>Confidence</th>
                            </tr>
                        </thead>
                        <tbody id="resultsTableBody">
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div class="card">
                <h2 style="margin-bottom: 20px;">ℹ️ About</h2>
                <p>This dashboard uses multiple AI models to analyze sentiment:</p>
                <ul style="margin: 15px 0 15px 30px; line-height: 1.8;">
                    <li><strong>Naïve Bayes:</strong> Probabilistic classifier based on Bayes' theorem</li>
                    <li><strong>Logistic Regression:</strong> Statistical model for binary/multiclass classification</li>
                    <li><strong>LSTM:</strong> Deep learning model using Long Short-Term Memory networks</li>
                </ul>
                <p><strong>Use Cases:</strong> Brand monitoring, customer feedback analysis, political campaign tracking, product review analysis</p>
            </div>
        </div>
        
        <script>
            async function analyzeText() {
                const text = document.getElementById('textInput').value;
                if (!text.trim()) {
                    alert('Please enter some text to analyze!');
                    return;
                }
                
                document.getElementById('loading').style.display = 'block';
                document.getElementById('results').style.display = 'none';
                
                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({text: text})
                    });
                    
                    const data = await response.json();
                    
                    // Update stats
                    document.getElementById('posCount').textContent = data.positive_count || 0;
                    document.getElementById('neuCount').textContent = data.neutral_count || 0;
                    document.getElementById('negCount').textContent = data.negative_count || 0;
                    
                    // Update charts
                    document.getElementById('pieChart').src = 'data:image/png;base64,' + data.pie_chart;
                    document.getElementById('barChart').src = 'data:image/png;base64,' + data.bar_chart;
                    
                    // Update table
                    const tbody = document.getElementById('resultsTableBody');
                    tbody.innerHTML = '';
                    data.results.forEach(result => {
                        const row = tbody.insertRow();
                        row.innerHTML = `
                            <td>${result.text.substring(0, 100)}${result.text.length > 100 ? '...' : ''}</td>
                            <td><span class="badge ${result.sentiment}">${result.sentiment}</span></td>
                            <td>${(result.confidence * 100).toFixed(1)}%</td>
                        `;
                    });
                    
                    document.getElementById('results').style.display = 'block';
                } catch (error) {
                    alert('Error analyzing text: ' + error.message);
                } finally {
                    document.getElementById('loading').style.display = 'none';
                }
            }
        </script>
    </body>
    </html>
    """
    return html

@app.route('/analyze', methods=['POST'])
def analyze():
    """API endpoint for sentiment analysis"""
    data = request.json
    texts = data.get('text', '').strip().split('\n')
    texts = [t.strip() for t in texts if t.strip()]
    
    if not texts:
        return jsonify({'error': 'No text provided'}), 400
    
    results = []
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    
    # Use best available model
    if model_store['best_model']:
        preprocessor = model_store['preprocessor']
        
        for text in texts:
            # Preprocess
            processed = preprocessor.preprocess(text)
            
            # Predict based on model type
            if model_store['best_model_name'] == 'lstm':
                sentiment, confidence = model_store['best_model'].predict([processed])
                sentiment = sentiment[0]
                confidence = confidence[0]
            else:
                # Traditional ML model
                extractor = model_store['extractor']
                features = extractor.transform([processed])
                prediction = model_store['best_model'].predict(features)[0]
                
                # Get probabilities if available
                if hasattr(model_store['best_model'], 'predict_proba'):
                    proba = model_store['best_model'].predict_proba(features)[0]
                    confidence = np.max(proba)
                else:
                    confidence = 0.8  # Default confidence
                
                sentiment = prediction
            
            results.append({
                'text': text,
                'sentiment': sentiment,
                'confidence': float(confidence)
            })
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
    else:
        # Fallback: Simple keyword-based analysis
        positive_words = ['love', 'amazing', 'great', 'excellent', 'good', 'best', 'fantastic', 'happy', 'awesome', 'wonderful']
        negative_words = ['hate', 'terrible', 'worst', 'bad', 'awful', 'disappointed', 'angry', 'sad', 'horrible', 'poor']
        
        for text in texts:
            text_lower = text.lower()
            pos_score = sum(1 for word in positive_words if word in text_lower)
            neg_score = sum(1 for word in negative_words if word in text_lower)
            
            if pos_score > neg_score:
                sentiment = 'positive'
                confidence = min(0.5 + pos_score * 0.1, 0.95)
            elif neg_score > pos_score:
                sentiment = 'negative'
                confidence = min(0.5 + neg_score * 0.1, 0.95)
            else:
                sentiment = 'neutral'
                confidence = 0.6
            
            results.append({
                'text': text,
                'sentiment': sentiment,
                'confidence': confidence
            })
            sentiment_counts[sentiment] += 1
    
    # Generate charts
    pie_chart = create_pie_chart(sentiment_counts)
    bar_chart = create_bar_chart(sentiment_counts)
    
    return jsonify({
        'results': results,
        'sentiment_counts': sentiment_counts,
        'pie_chart': pie_chart,
        'bar_chart': bar_chart,
        'positive_count': sentiment_counts['positive'],
        'neutral_count': sentiment_counts['neutral'],
        'negative_count': sentiment_counts['negative']
    })

def train_and_save_models():
    """Train all models and save them"""
    print("\n" + "="*60)
    print("🚀 TRAINING SENTIMENT ANALYSIS MODELS")
    print("="*60)
    
    # Download NLTK data
    download_nltk_data()
    
    # Load data
    loader = DataLoader()
    df = loader.load_twitter_data()
    
    # Preprocess
    preprocessor = TextPreprocessor()
    df = preprocessor.preprocess_dataframe(df, text_column='text')
    model_store['preprocessor'] = preprocessor
    
    # Feature extraction
    extractor = FeatureExtractor(method='tfidf', max_features=5000)
    X = extractor.fit_transform(df['processed_text'])
    y = df['sentiment'].values
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Dataset: {len(y_train)} train, {len(y_test)} test samples")
    
    # Train traditional models
    traditional = TraditionalModels()
    traditional.train_naive_bayes(X_train, y_train, X_test, y_test)
    traditional.train_logistic_regression(X_train, y_train, X_test, y_test)
    traditional.train_svm(X_train, y_train, X_test, y_test)
    
    model_store['traditional_models'] = traditional
    model_store['extractor'] = extractor
    
    # Get best traditional model
    best_name, best_model = traditional.get_best_model()
    model_store['best_model_name'] = best_name
    model_store['best_model'] = best_model
    
    print(f"\n🏆 Best Traditional Model: {best_name}")
    
    # Train LSTM
    print("\n🧠 Training LSTM (this may take a few minutes)...")
    lstm = LSTMModel(max_words=5000, max_len=50)
    X_lstm, y_lstm = lstm.prepare_sequences(df['processed_text'], df['sentiment'])
    
    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
        X_lstm, y_lstm, test_size=0.2, random_state=42
    )
    
    lstm.build_model(num_classes=y_lstm.shape[1])
    lstm.train(X_train_l, y_train_l, X_test_l, y_test_l, epochs=10, batch_size=32)
    
    model_store['lstm_model'] = lstm
    
    # Save models
    os.makedirs('../models', exist_ok=True)
    extractor.save('../models/feature_extractor.pkl')
    traditional.save_model(best_name, f'../models/{best_name}.pkl')
    lstm.save('../models/lstm_model.h5', '../models/lstm_tokenizer.pkl')
    
    print("\n✅ All models trained and saved!")
    print("="*60)

if __name__ == '__main__':
    # Check if models exist, if not train them
    if not os.path.exists('../models/feature_extractor.pkl'):
        print("⚠️  No trained models found. Training now...")
        train_and_save_models()
    else:
        print("📂 Loading existing models...")
        # Load preprocessor
        with open('../models/preprocessor.pkl', 'rb') as f:
            model_store['preprocessor'] = pickle.load(f)
        
        # Load feature extractor
        extractor = FeatureExtractor(method='tfidf')
        extractor.load('../models/feature_extractor.pkl')
        model_store['extractor'] = extractor
        
        # Load best traditional model
        import glob
        model_files = glob.glob('../models/*.pkl')
        for mf in model_files:
            if 'feature_extractor' not in mf and 'tokenizer' not in mf:
                with open(mf, 'rb') as f:
                    model_store['best_model'] = pickle.load(f)
                model_store['best_model_name'] = os.path.basename(mf).replace('.pkl', '')
                break
    
    print("\n🌐 Starting Flask server...")
    print("👉 Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)