
import numpy as np
import pandas as pd
import pickle
import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any

# Traditional ML
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve
)

# Deep Learning (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import (
        Embedding, LSTM, GRU, Dense, Dropout, Bidirectional,
        GlobalMaxPooling1D, GlobalAveragePooling1D, Conv1D, MaxPooling1D,
        Flatten, BatchNormalization, Attention, MultiHeadAttention,
        LayerNormalization, Input, SpatialDropout1D, Lambda
    )
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical, plot_model
    from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
        TensorBoard, CSVLogger
    )
    from tensorflow.keras.optimizers import Adam, RMSprop
    from tensorflow.keras.regularizers import l2
    TF_AVAILABLE = True
except ImportError:
    tf = None
    Sequential = Model = load_model = None
    Embedding = LSTM = GRU = Dense = Dropout = Bidirectional = None
    GlobalMaxPooling1D = GlobalAveragePooling1D = Conv1D = MaxPooling1D = None
    Flatten = BatchNormalization = Attention = MultiHeadAttention = None
    LayerNormalization = Input = SpatialDropout1D = Lambda = None
    Tokenizer = pad_sequences = to_categorical = plot_model = None
    EarlyStopping = ModelCheckpoint = ReduceLROnPlateau = TensorBoard = CSVLogger = None
    Adam = RMSprop = l2 = None
    TF_AVAILABLE = False

class BaseModel(ABC):
    """Abstract base class for all models"""
    
    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod
    def evaluate(self, X_test, y_test):
        pass
    
    @abstractmethod
    def save(self, filepath):
        pass
    
    @abstractmethod
    def load(self, filepath):
        pass


class ModelResults:
    """Store and compare model results"""
    
    def __init__(self):
        self.results = {}
    
    def add_result(self, model_name: str, metrics: Dict):
        self.results[model_name] = {
            'accuracy': metrics.get('accuracy', 0),
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'f1': metrics.get('f1', 0),
            'training_time': metrics.get('training_time', 0),
            'params': metrics.get('params', {})
        }
    
    def get_best_model(self, metric='f1'):
        if not self.results:
            return None, None
        
        best = max(self.results.items(), key=lambda x: x[1].get(metric, 0))
        return best[0], best[1]
    
    def compare_all(self):
        print("\n" + "="*70)
        print("📊 MODEL COMPARISON")
        print("="*70)
        print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-"*70)
        
        for name, metrics in sorted(self.results.items(), 
                                   key=lambda x: x[1]['f1'], reverse=True):
            print(f"{name:<25} {metrics['accuracy']:<10.4f} "
                  f"{metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                  f"{metrics['f1']:<10.4f}")
        
        print("="*70)
        best_name, best_metrics = self.get_best_model('f1')
        print(f"🏆 Best Model: {best_name} (F1: {best_metrics['f1']:.4f})")
        return best_name


class TraditionalModels:
    """Collection of traditional ML models with hyperparameter tuning"""
    
    def __init__(self):
        self.models = {}
        self.results = ModelResults()
        self.best_model = None
        self.best_model_name = None
    
    def _calculate_metrics(self, y_true, y_pred, y_proba=None):
        """Calculate comprehensive metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Add per-class metrics
        report = classification_report(y_true, y_pred, output_dict=True)
        metrics['per_class'] = report
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        # ROC-AUC if binary and probabilities available
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            except:
                pass
        
        return metrics
    
    def train_naive_bayes(self, X_train, y_train, X_test, y_test, tune=False):
        """Train Naive Bayes with optional hyperparameter tuning"""
        print("\n🎓 Training Naive Bayes...")
        
        if tune:
            params = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]}
            grid = GridSearchCV(MultinomialNB(), params, cv=5, scoring='f1_weighted')
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            print(f"   Best alpha: {grid.best_params_['alpha']}")
        else:
            model = MultinomialNB(alpha=1.0)
            model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_proba)
        
        self.models['naive_bayes'] = model
        self.results.add_result('naive_bayes', metrics)
        
        self._print_metrics("Naive Bayes", metrics)
        return model, metrics
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test, tune=False):
        """Train Logistic Regression with optional tuning"""
        print("\n🎓 Training Logistic Regression...")
        
        if tune:
            params = {
                'C': [0.1, 1.0, 10.0],
                'solver': ['lbfgs', 'liblinear']
            }
            grid = GridSearchCV(
                LogisticRegression(max_iter=1000, multi_class='auto'),
                params, cv=5, scoring='f1_weighted', n_jobs=-1
            )
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            print(f"   Best params: {grid.best_params_}")
        else:
            model = LogisticRegression(
                C=1.0, max_iter=1000, solver='lbfgs', multi_class='auto'
            )
            model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        metrics = self._calculate_metrics(y_test, y_pred, y_proba)
        
        self.models['logistic_regression'] = model
        self.results.add_result('logistic_regression', metrics)
        
        self._print_metrics("Logistic Regression", metrics)
        return model, metrics
    
    def train_svm(self, X_train, y_train, X_test, y_test, tune=False, kernel='linear'):
        """Train SVM with optional tuning"""
        print(f"\n🎓 Training SVM ({kernel} kernel)...")
        
        if kernel == 'linear':
            model = SVC(kernel='linear', C=1.0, probability=True)
        else:
            model = SVC(kernel=kernel, C=1.0, probability=True)
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
        elif hasattr(model, 'decision_function'):
            # Convert decision function to probabilities using softmax
            decisions = model.decision_function(X_test)
            if len(decisions.shape) == 1:
                decisions = np.column_stack([-decisions, decisions])
            exp_decisions = np.exp(decisions - np.max(decisions, axis=1, keepdims=True))
            y_proba = exp_decisions / np.sum(exp_decisions, axis=1, keepdims=True)
        else:
            y_proba = None
        
        metrics = self._calculate_metrics(y_test, y_pred, y_proba)
        
        model_name = f'svm_{kernel}' if kernel != 'linear' else 'svm'
        self.models[model_name] = model
        self.results.add_result(model_name, metrics)
        
        self._print_metrics(f"SVM ({kernel})", metrics)
        return model, metrics
    
    def train_random_forest(self, X_train, y_train, X_test, y_test, tune=False):
        """Train Random Forest with optional tuning"""
        print("\n🎓 Training Random Forest...")
        
        if tune:
            params = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
            grid = GridSearchCV(
                RandomForestClassifier(random_state=42),
                params, cv=3, scoring='f1_weighted', n_jobs=-1
            )
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
        else:
            model = RandomForestClassifier(
                n_estimators=100, max_depth=None, random_state=42, n_jobs=-1
            )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        metrics = self._calculate_metrics(y_test, y_pred, y_proba)
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            metrics['feature_importance'] = model.feature_importances_.tolist()
        
        self.models['random_forest'] = model
        self.results.add_result('random_forest', metrics)
        
        self._print_metrics("Random Forest", metrics)
        return model, metrics
    
    def train_gradient_boosting(self, X_train, y_train, X_test, y_test):
        """Train Gradient Boosting classifier"""
        print("\n🎓 Training Gradient Boosting...")
        
        model = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        metrics = self._calculate_metrics(y_test, y_pred, y_proba)
        
        self.models['gradient_boosting'] = model
        self.results.add_result('gradient_boosting', metrics)
        
        self._print_metrics("Gradient Boosting", metrics)
        return model, metrics
    
    def _print_metrics(self, name, metrics):
        """Print formatted metrics"""
        print(f"\n📊 {name} Results:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1-Score:  {metrics['f1']:.4f}")
    
    def get_best_model(self):
        """Get best performing model"""
        best_name, best_metrics = self.results.get_best_model('f1')
        self.best_model_name = best_name
        self.best_model = self.models.get(best_name)
        return best_name, self.best_model
    
    def cross_validate(self, X, y, model_name='logistic_regression', cv=5):
        """Perform cross-validation"""
        model = self.models.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found. Train it first.")
        
        print(f"\n🔍 Cross-validating {model_name}...")
        scores = cross_val_score(model, X, y, cv=StratifiedKFold(cv), scoring='f1_weighted')
        
        print(f"   CV Scores: {scores}")
        print(f"   Mean F1: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores
    
    def save_model(self, model_name, filepath):
        """Save specific model"""
        model = self.models.get(model_name)
        if model:
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            print(f"💾 Saved {model_name} to {filepath}")
    
    def load_model(self, model_name, filepath):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            self.models[model_name] = pickle.load(f)
        print(f"📂 Loaded {model_name} from {filepath}")


class LSTMModel(BaseModel):
    """LSTM Deep Learning Model with advanced architecture options"""
    
    def __init__(self, max_words=10000, max_len=100, embedding_dim=128):
        if not TF_AVAILABLE:
            raise ImportError('TensorFlow is required for LSTMModel. Install tensorflow to use this model.')
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.tokenizer = None
        self.model = None
        self.label_mapping = None
        self.reverse_mapping = None
        self.history = None
        self.classes = None
    
    def prepare_sequences(self, texts, labels=None, fit_tokenizer=True):
        """Convert texts to sequences"""
        print("📝 Preparing sequences...")
        
        # Create tokenizer
        if fit_tokenizer or self.tokenizer is None:
            self.tokenizer = Tokenizer(
                num_words=self.max_words,
                oov_token='<OOV>',
                filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
            )
            self.tokenizer.fit_on_texts(texts)
        
        # Convert to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        
        # Handle labels
        if labels is not None:
            self.classes = sorted(set(labels))
            self.label_mapping = {label: idx for idx, label in enumerate(self.classes)}
            self.reverse_mapping = {idx: label for label, idx in self.label_mapping.items()}
            
            y_encoded = [self.label_mapping[label] for label in labels]
            y = to_categorical(y_encoded, num_classes=len(self.classes))
            
            print(f"   Classes: {self.classes}")
            print(f"   X shape: {X.shape}, y shape: {y.shape}")
            return X, y
        
        return X
    
    def build_simple_lstm(self, num_classes):
        """Build simple LSTM model"""
        model = Sequential([
            Embedding(self.max_words, self.embedding_dim, input_length=self.max_len),
            SpatialDropout1D(0.2),
            LSTM(128, dropout=0.2, recurrent_dropout=0.2),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        return model
    
    def build_bidirectional_lstm(self, num_classes):
        """Build bidirectional LSTM with attention"""
        inputs = Input(shape=(self.max_len,))
        
        # Embedding
        x = Embedding(self.max_words, self.embedding_dim)(inputs)
        x = SpatialDropout1D(0.2)(x)
        
        # Bidirectional LSTM layers
        x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(x)
        x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2))(x)
        
        # Attention mechanism (simplified)
        attention = Dense(1, activation='tanh')(x)
        attention = Flatten()(attention)
        attention = tf.keras.layers.Activation('softmax')(attention)
        attention = tf.keras.layers.RepeatVector(128)(attention)
        attention = tf.keras.layers.Permute([2, 1])(attention)
        
        # Apply attention
        sent_representation = tf.keras.layers.multiply([x, attention])
        sent_representation = Lambda(lambda xin: tf.keras.backend.sum(xin, axis=1))(sent_representation)
        
        # Dense layers
        x = Dense(64, activation='relu')(sent_representation)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def build_cnn_lstm(self, num_classes):
        """Build CNN + LSTM hybrid"""
        model = Sequential([
            Embedding(self.max_words, self.embedding_dim, input_length=self.max_len),
            
            # CNN layers for local feature extraction
            Conv1D(128, 5, activation='relu'),
            MaxPooling1D(2),
            Conv1D(64, 3, activation='relu'),
            MaxPooling1D(2),
            
            # LSTM for sequence modeling
            LSTM(128, dropout=0.2, recurrent_dropout=0.2),
            
            # Dense layers
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        return model
    
    def build_model(self, num_classes, architecture='bidirectional'):
        """Build model with specified architecture"""
        print(f"\n🏗️  Building {architecture} LSTM model...")
        
        if architecture == 'simple':
            self.model = self.build_simple_lstm(num_classes)
        elif architecture == 'bidirectional':
            self.model = self.build_bidirectional_lstm(num_classes)
        elif architecture == 'cnn_lstm':
            self.model = self.build_cnn_lstm(num_classes)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Compile
        optimizer = Adam(learning_rate=0.001)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        print(f"   Total parameters: {self.model.count_params():,}")
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32, 
              use_callbacks=True, model_name='best_lstm'):
        """Train the model"""
        print(f"\n🚀 Training LSTM for max {epochs} epochs...")
        
        callbacks = []
        
        if use_callbacks:
            # Early stopping
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stop)
            
            # Model checkpoint
            checkpoint = ModelCheckpoint(
                f'{model_name}.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
            callbacks.append(checkpoint)
            
            # Learning rate reduction
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
            callbacks.append(reduce_lr)
            
            # CSV logging
            csv_logger = CSVLogger('training_log.csv', append=True)
            callbacks.append(csv_logger)
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Find best epoch
        best_epoch = np.argmax(self.history.history['val_accuracy'])
        best_acc = self.history.history['val_accuracy'][best_epoch]
        print(f"\n✅ Best validation accuracy: {best_acc:.4f} at epoch {best_epoch + 1}")
        
        return self.history
    
    def predict(self, texts, return_confidence=True):
        """Make predictions on new texts"""
        # Prepare sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_len, padding='post')
        
        # Predict
        predictions = self.model.predict(X, verbose=0)
        
        # Get labels and confidence
        predicted_indices = np.argmax(predictions, axis=1)
        predicted_labels = [self.reverse_mapping[idx] for idx in predicted_indices]
        
        if return_confidence:
            confidence = np.max(predictions, axis=1)
            return predicted_labels, confidence
        
        return predicted_labels
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        print("\n📊 LSTM Evaluation:")
        loss, accuracy, precision, recall = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"   Loss:      {loss:.4f}")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {2 * (precision * recall) / (precision + recall):.4f}")
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }
    
    def plot_history(self, save_path=None):
        """Plot training history"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Precision
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Train')
            axes[1, 0].plot(self.history.history['val_precision'], label='Validation')
            axes[1, 0].set_title('Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
        
        # Recall
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Train')
            axes[1, 1].plot(self.history.history['val_recall'], label='Validation')
            axes[1, 1].set_title('Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        plt.close()
    
    def save(self, model_path, tokenizer_path=None):
        """Save model and tokenizer"""
        # Save Keras model
        self.model.save(model_path)
        
        # Save tokenizer and mappings
        if tokenizer_path:
            with open(tokenizer_path, 'wb') as f:
                pickle.dump({
                    'tokenizer': self.tokenizer,
                    'label_mapping': self.label_mapping,
                    'reverse_mapping': self.reverse_mapping,
                    'max_len': self.max_len,
                    'classes': self.classes
                }, f)
        
        print(f"💾 LSTM model saved: {model_path}")
    
    def load(self, model_path, tokenizer_path):
        """Load model and tokenizer"""
        # Load Keras model
        self.model = load_model(model_path)
        
        # Load tokenizer and mappings
        with open(tokenizer_path, 'rb') as f:
            data = pickle.load(f)
            self.tokenizer = data['tokenizer']
            self.label_mapping = data['label_mapping']
            self.reverse_mapping = data['reverse_mapping']
            self.max_len = data['max_len']
            self.classes = data['classes']
        
        print(f"📂 LSTM model loaded: {model_path}")


class EnsembleModel:
    """Ensemble of multiple models for improved predictions"""
    
    def __init__(self, models=None, weights=None):
        self.models = models or {}
        self.weights = weights or {}
        self.voting = 'soft'  # 'hard' or 'soft'
    
    def add_model(self, name, model, weight=1.0):
        """Add a model to the ensemble"""
        self.models[name] = model
        self.weights[name] = weight
    
    def predict(self, X, preprocessor=None, extractor=None):
        """Make ensemble prediction"""
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            if name == 'lstm':
                # LSTM prediction
                pred, conf = model.predict(X, return_confidence=True)
                predictions[name] = pred
                probabilities[name] = conf
            else:
                # Traditional model
                if extractor:
                    features = extractor.transform(X)
                else:
                    features = X
                
                pred = model.predict(features)
                predictions[name] = pred
                
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features)
                    probabilities[name] = np.max(proba, axis=1)
        
        # Weighted voting
        if self.voting == 'soft' and probabilities:
            # Average probabilities (simplified)
            # In practice, you'd need to align class probabilities
            final_pred = predictions[max(self.weights, key=self.weights.get)]
        else:
            # Hard voting - majority vote
            from scipy import stats
            all_preds = np.array(list(predictions.values()))
            final_pred = stats.mode(all_preds, axis=0)[0].flatten()
        
        return final_pred


if __name__ == "__main__":
    print("✅ Models module loaded successfully")
    print("Available models:")
    print("  - Naive Bayes")
    print("  - Logistic Regression")
    print("  - SVM (Linear/RBF)")
    print("  - Random Forest")
    print("  - Gradient Boosting")
    print("  - LSTM (Simple/Bidirectional/CNN-LSTM)")
    print("  - Ensemble")


