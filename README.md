📊 Sentiment Analysis of Amazon Product Reviews
📌 Project Overview
This project focuses on automated sentiment classification of Amazon product reviews. The goal is to build a machine learning system that categorizes reviews into positive, neutral, or negative sentiments, enabling e-commerce platforms to monitor customer feedback in real time.

Working Dashboard Vedio Link - https://youtu.be/Zb4HXzh-gXI

🔑 Business Value
- Enhanced customer experience monitoring
- Data-driven product improvement recommendations
- Competitive intelligence through sentiment trend analysis
- Reduced manual review processing costs

📂 Dataset
- Source: Datafiniti Amazon Product Reviews
- Size: 1,597 reviews → 1,176 after cleaning
- Features: reviews.text, reviews.rating
Class Distribution
- Positive: 83.1% (977 reviews)
- Neutral: 10.5% (123 reviews)
- Negative: 6.4% (76 reviews)

⚙️ Methods
Preprocessing Pipeline
- Text cleaning (remove special chars, HTML, URLs)
- Tokenization
- Lemmatization
- Stopword removal (negations preserved)
- TF-IDF vectorization (5,000 features, 1–2 grams)
Models Evaluated
Naïve Bayes         - Accuracy(82.3%),F1-Score(81.9%)
Logistic Regression - Accuracy(88.6%),F1-Score(88.7%)
LSTM (BiLSTM)       - Accuracy(85.2%),F1-Score(84.8%)


Deep Learning Architecture
Embedding(128) → BiLSTM(64) → Dense(64) → Softmax

📈 Results & Insights
- Best Model: Logistic Regression (88.6% accuracy, 88.7% F1-score)
- Feature Importance: Positive words like excellent, amazing, great, love, perfect; Negative words like terrible, awful, horrible, waste, disappointed
- Prediction Confidence: Model outputs confidence scores for uncertainty quantification

⚠️ Limitations
- Small dataset size (1,176 reviews)
- Class imbalance (83% positive reviews)
- Domain specificity (Amazon-only data)
- Sarcasm detection challenges

🚀 Future Improvements
- Integration of BERT/transformer-based models
- Aspect-based sentiment analysis
- Multi-language support
- Active learning for dataset expansion
- Ensemble methods for robustness

