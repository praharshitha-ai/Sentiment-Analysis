import os
import io
import base64
import json
from datetime import datetime
from collections import Counter, defaultdict
import threading
import time

import numpy as np
import pandas as pd

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Wedge, Rectangle
import seaborn as sns
from wordcloud import WordCloud

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'

class DashboardAnalytics:
    """
    Advanced analytics and visualization for sentiment data
    """
    
    def __init__(self):
        self.color_scheme = {
            'positive': '#28a745',
            'negative': '#dc3545', 
            'neutral': '#ffc107',
            'background': '#f8f9fa',
            'text': '#333333'
        }
        self.history = []
        self.stats_cache = {}
    
    # ==================== CHART GENERATION ====================
    
    def create_pie_chart(self, sentiment_counts, title="Sentiment Distribution", 
                        donut=True, show_legend=True):
        """Create pie chart with optional donut style"""
        fig, ax = plt.subplots(figsize=(10, 8), facecolor=self.color_scheme['background'])
        
        colors = [self.color_scheme[k] for k in ['positive', 'neutral', 'negative'] 
                 if k in sentiment_counts]
        labels = list(sentiment_counts.keys())
        sizes = list(sentiment_counts.values())
        
        # Create pie
        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=labels if not donut else None,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 12, 'weight': 'bold'},
            pctdistance=0.75 if donut else 0.5
        )
        
        # Style percentages
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(11)
            autotext.set_weight('bold')
        
        # Add donut center
        if donut:
            centre_circle = plt.Circle((0, 0), 0.50, fc=self.color_scheme['background'])
            ax.add_artist(centre_circle)
            
            # Total in center
            total = sum(sizes)
            ax.text(0, 0.05, str(total), ha='center', va='center', 
                   fontsize=24, weight='bold', color=self.color_scheme['text'])
            ax.text(0, -0.15, 'Total', ha='center', va='center', 
                   fontsize=12, color='gray')
        
        # Title
        ax.set_title(title, fontsize=16, weight='bold', pad=20)
        
        # Legend
        if show_legend and donut:
            ax.legend(wedges, labels, title="Sentiments", loc="center left", 
                     bbox_to_anchor=(1, 0, 0.5, 1))
        
        return self._fig_to_base64(fig)
    
    def create_bar_chart(self, sentiment_counts, title="Sentiment Counts", 
                        horizontal=False, show_values=True):
        """Create bar chart"""
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=self.color_scheme['background'])
        
        labels = list(sentiment_counts.keys())
        values = list(sentiment_counts.values())
        colors = [self.color_scheme.get(l, '#6c757d') for l in labels]
        
        if horizontal:
            bars = ax.barh(labels, values, color=colors, edgecolor='white', linewidth=2)
            # Add value labels
            if show_values:
                for i, (bar, val) in enumerate(zip(bars, values)):
                    ax.text(val + max(values)*0.01, i, str(val), 
                           va='center', fontsize=11, weight='bold')
        else:
            bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=2)
            # Add value labels
            if show_values:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom',
                           fontsize=11, weight='bold')
        
        ax.set_title(title, fontsize=16, weight='bold', pad=20)
        ax.set_facecolor(self.color_scheme['background'])
        
        if horizontal:
            ax.set_xlabel('Count', fontsize=12)
        else:
            ax.set_ylabel('Count', fontsize=12)
        
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        return self._fig_to_base64(fig)
    
    def create_trend_chart(self, timestamps, sentiments, title="Sentiment Over Time"):
        """Create time-series sentiment trend"""
        fig, ax = plt.subplots(figsize=(12, 6), facecolor=self.color_scheme['background'])
        
        # Group by time and sentiment
        df = pd.DataFrame({'timestamp': timestamps, 'sentiment': sentiments})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Create time buckets (hourly)
        df['hour'] = df['timestamp'].dt.floor('H')
        trend = df.groupby(['hour', 'sentiment']).size().unstack(fill_value=0)
        
        # Plot lines
        for sentiment in ['positive', 'neutral', 'negative']:
            if sentiment in trend.columns:
                ax.plot(trend.index, trend[sentiment], 
                       marker='o', linewidth=2, label=sentiment,
                       color=self.color_scheme[sentiment])
        
        ax.set_title(title, fontsize=16, weight='bold', pad=20)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
    
    def create_confidence_distribution(self, confidences, sentiments, 
                                      title="Confidence Distribution"):
        """Create confidence score distribution by sentiment"""
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=self.color_scheme['background'])
        
        data = pd.DataFrame({'confidence': confidences, 'sentiment': sentiments})
        
        for sentiment in ['positive', 'neutral', 'negative']:
            subset = data[data['sentiment'] == sentiment]['confidence']
            if len(subset) > 0:
                ax.hist(subset, bins=20, alpha=0.6, label=sentiment,
                       color=self.color_scheme[sentiment], edgecolor='white')
        
        ax.set_title(title, fontsize=16, weight='bold', pad=20)
        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend()
        ax.set_facecolor(self.color_scheme['background'])
        
        return self._fig_to_base64(fig)
    
    def create_wordcloud(self, texts, sentiment=None, title=None, 
                        width=800, height=400):
        """Create word cloud from texts"""
        if not texts:
            return None
        
        # Join all texts
        text = ' '.join(texts)
        
        # Color function based on sentiment
        def color_func(word, *args, **kwargs):
            if sentiment == 'positive':
                return '#28a745'
            elif sentiment == 'negative':
                return '#dc3545'
            elif sentiment == 'neutral':
                return '#ffc107'
            return '#333333'
        
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color='white',
            colormap='viridis',
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        
        if title:
            ax.set_title(title, fontsize=16, weight='bold', pad=10)
        
        return self._fig_to_base64(fig)
    
    def create_comparison_chart(self, model_results, title="Model Comparison"):
        """Compare multiple models' performance"""
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=self.color_scheme['background'])
        
        models = list(model_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [model_results[m].get(metric, 0) for m in models]
            ax.bar(x + i*width, values, width, label=metric.capitalize())
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=16, weight='bold', pad=20)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        return self._fig_to_base64(fig)
    
    def create_gauge_chart(self, percentage, sentiment, title="Sentiment Score"):
        """Create gauge chart for single prediction"""
        fig, ax = plt.subplots(figsize=(8, 4), facecolor='white')
        
        color = self.color_scheme.get(sentiment, '#6c757d')
        
        # Create gauge background
        theta = np.linspace(0, np.pi, 100)
        
        # Background arc (gray)
        ax.fill_between(np.cos(theta), np.sin(theta), 0, 
                       alpha=0.1, color='gray')
        
        # Value arc (colored)
        value_theta = theta[:int(len(theta) * percentage / 100)]
        ax.fill_between(np.cos(value_theta), np.sin(value_theta), 0,
                       alpha=0.8, color=color)
        
        # Needle
        needle_angle = np.pi * (1 - percentage / 100)
        ax.annotate('', xy=(np.cos(needle_angle), np.sin(needle_angle)),
                   xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', color='black', lw=3))
        
        # Text
        ax.text(0, -0.3, f'{percentage:.1f}%', ha='center', va='center',
               fontsize=28, weight='bold', color=color)
        ax.text(0, -0.6, sentiment.upper(), ha='center', va='center',
               fontsize=14, weight='bold', color=color)
        
        # Labels
        ax.text(-1, -0.1, '0%', ha='center', fontsize=10, color='gray')
        ax.text(1, -0.1, '100%', ha='center', fontsize=10, color='gray')
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.8, 1.2)
        ax.axis('off')
        
        return self._fig_to_base64(fig)
    
    def create_heatmap(self, confusion_matrix, labels, title="Confusion Matrix"):
        """Create confusion matrix heatmap"""
        fig, ax = plt.subplots(figsize=(8, 6), facecolor=self.color_scheme['background'])
        
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels, ax=ax)
        
        ax.set_title(title, fontsize=16, weight='bold', pad=20)
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        
        return self._fig_to_base64(fig)
    
    # ==================== UTILITY METHODS ====================
    
    def _fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 string"""
        buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                   facecolor=fig.get_facecolor())
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode('utf-8')
    
    def generate_report(self, texts, predictions, confidences, model_name):
        """Generate comprehensive HTML report"""
        sentiments = [p['sentiment'] if isinstance(p, dict) else p 
                     for p in predictions]
        
        # Calculate statistics
        stats = {
            'total': len(texts),
            'positive': sentiments.count('positive'),
            'neutral': sentiments.count('neutral'),
            'negative': sentiments.count('negative'),
            'avg_confidence': np.mean(confidences) if confidences else 0
        }
        
        # Generate charts
        sentiment_counts = {
            'positive': stats['positive'],
            'neutral': stats['neutral'],
            'negative': stats['negative']
        }
        
        charts = {
            'pie': self.create_pie_chart(sentiment_counts),
            'bar': self.create_bar_chart(sentiment_counts),
            'confidence': self.create_confidence_distribution(confidences, sentiments)
        }
        
        # Generate word clouds
        pos_texts = [t for t, s in zip(texts, sentiments) if s == 'positive']
        neg_texts = [t for t, s in zip(texts, sentiments) if s == 'negative']
        
        if pos_texts:
            charts['wordcloud_pos'] = self.create_wordcloud(pos_texts, 'positive')
        if neg_texts:
            charts['wordcloud_neg'] = self.create_wordcloud(neg_texts, 'negative')
        
        return {
            'stats': stats,
            'charts': charts,
            'timestamp': datetime.now().isoformat(),
            'model': model_name
        }


class RealtimeDashboard:
    """
    Real-time dashboard with live updates
    """
    
    def __init__(self, update_interval=5):
        self.analytics = DashboardAnalytics()
        self.update_interval = update_interval
        self.data_buffer = []
        self.is_running = False
        self.thread = None
        self.callbacks = []
    
    def start(self):
        """Start real-time updates"""
        self.is_running = True
        self.thread = threading.Thread(target=self._update_loop)
        self.thread.daemon = True
        self.thread.start()
        print("✅ Real-time dashboard started")
    
    def stop(self):
        """Stop real-time updates"""
        self.is_running = False
        if self.thread:
            self.thread.join
