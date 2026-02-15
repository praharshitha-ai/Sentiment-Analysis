import unittest
import sys
import os
import time
import pickle
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

# Import test utilities
from tests import (
    TEST_CONFIG, SAMPLE_TEXTS, get_test_data, 
    get_sample_dataframe, skip_slow_tests, require_model_files
)

# Import project modules
from src.preprocess import TextPreprocessor, download_nltk_data
from src.features import SimpleFeatureExtractor, AdvancedFeatureExtractor
from src.models import (
    TraditionalModels, LSTMModel, EnsembleModel, 
    ModelResults, BaseModel
)

# Try to import advanced models
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')  # Disable GPU for tests
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available, skipping LSTM tests")


class TestModelResults(unittest.TestCase):
    """Test ModelResults class"""
    
    def setUp(self):
        self.results = ModelResults()
    
    def test_add_result(self):
        """Test adding results"""
        metrics = {
            'accuracy': 0.85,
            'precision': 0.84,
            'recall': 0.86,
            'f1': 0.85
        }
        self.results.add_result('test_model', metrics)
        
        self.assertIn('test_model', self.results.results)
        self.assertEqual(self.results.results['test_model']['f1'], 0.85)
    
    def test_get_best_model(self):
        """Test getting best model"""
        # Add multiple models
        self.results.add_result('model_a', {'f1': 0.80, 'accuracy': 0.81})
        self.results.add_result('model_b', {'f1': 0.90, 'accuracy': 0.89})
        self.results.add_result('model_c', {'f1': 0.85, 'accuracy': 0.86})
        
        best_name, best_metrics = self.results.get_best_model('f1')
        
        self.assertEqual(best_name, 'model_b')
        self.assertEqual(best_metrics['f1'], 0.90)
    
    def test_compare_all(self):
        """Test comparison method"""
        self.results.add_result('model_a', {'f1': 0.80, 'accuracy': 0.81, 'precision': 0.82, 'recall': 0.83})
        self.results.add_result('model_b', {'f1': 0.90, 'accuracy': 0.89, 'precision': 0.88, 'recall': 0.87})
        
        # Should not raise exception
        try:
            self.results.compare_all()
        except Exception as e:
            self.fail(f"compare_all() raised {e}")


class TestTraditionalModels(unittest.TestCase):
    """Test traditional ML models"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data once for all tests"""
        download_nltk_data()
        
        # Create sample data
        cls.df = get_sample_dataframe()
        
        # Preprocess
        cls.preprocessor = TextPreprocessor()
        cls.df = cls.preprocessor.preprocess_dataframe(cls.df)
        
        # Extract features
        cls.extractor = SimpleFeatureExtractor(method='tfidf', max_features=100)
        cls.X = cls.extractor.fit_transform(cls.df['processed_text'])
        cls.y = cls.df['sentiment'].values
        
        # Split data
        from sklearn.model_selection import train_test_split
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.X, cls.y, test_size=0.3, random_state=TEST_CONFIG['random_seed']
        )
        
        cls.models = TraditionalModels()
    
    def test_train_naive_bayes(self):
        """Test Naive Bayes training"""
        model, metrics = self.models.train_naive_bayes(
            self.X_train, self.y_train, 
            self.X_test, self.y_test
        )
        
        self.assertIsNotNone(model)
        self.assertIn('naive_bayes', self.models.models)
        self.assertGreater(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1.0)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
    
    def test_train_logistic_regression(self):
        """Test Logistic Regression training"""
        model, metrics = self.models.train_logistic_regression(
            self.X_train, self.y_train,
            self.X_test, self.y_test
        )
        
        self.assertIsNotNone(model)
        self.assertIn('logistic_regression', self.models.models)
        self.assertGreater(metrics['accuracy'], 0)
    
    def test_train_svm(self):
        """Test SVM training"""
        model, metrics = self.models.train_svm(
            self.X_train, self.y_train,
            self.X_test, self.y_test
        )
        
        self.assertIsNotNone(model)
        self.assertIn('svm', self.models.models)
        self.assertGreater(metrics['accuracy'], 0)
    
    def test_train_random_forest(self):
        """Test Random Forest training"""
        model, metrics = self.models.train_random_forest(
            self.X_train, self.y_train,
            self.X_test, self.y_test
        )
        
        self.assertIsNotNone(model)
        self.assertIn('random_forest', self.models.models)
        self.assertGreater(metrics['accuracy'], 0)
    
    def test_get_best_model(self):
        """Test getting best model"""
        # Train multiple models
        self.models.train_naive_bayes(self.X_train, self.y_train, self.X_test, self.y_test)
        self.models.train_logistic_regression(self.X_train, self.y_train, self.X_test, self.y_test)
        
        best_name, best_model = self.models.get_best_model()
        
        self.assertIsNotNone(best_name)
        self.assertIsNotNone(best_model)
        self.assertIn(best_name, self.models.models)
    
    def test_cross_validate(self):
        """Test cross-validation"""
        self.models.train_logistic_regression(
            self.X_train, self.y_train,
            self.X_test, self.y_test
        )
        
        scores = self.models.cross_validate(self.X, self.y, 'logistic_regression', cv=3)
        
        self.assertEqual(len(scores), 3)
        self.assertTrue(all(0 <= score <= 1 for score in scores))
    
    def test_model_saving_loading(self):
        """Test saving and loading models"""
        # Train and save
        self.models.train_naive_bayes(self.X_train, self.y_train, self.X_test, self.y_test)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            self.models.save_model('naive_bayes', temp_path)
            self.assertTrue(os.path.exists(temp_path))
            
            # Load into new instance
            new_models = TraditionalModels()
            new_models.load_model('naive_bayes_loaded', temp_path)
            
            self.assertIn('naive_bayes_loaded', new_models.models)
            
            # Test prediction
            pred = new_models.models['naive_bayes_loaded'].predict(self.X_test[:1])
            self.assertEqual(len(pred), 1)
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_prediction_consistency(self):
        """Test that predictions are consistent"""
        self.models.train_logistic_regression(
            self.X_train, self.y_train,
            self.X_test, self.y_test
        )
        
        model = self.models.models['logistic_regression']
        
        # Predict same input multiple times
        pred1 = model.predict(self.X_test[:5])
        pred2 = model.predict(self.X_test[:5])
        pred3 = model.predict(self.X_test[:5])
        
        np.testing.assert_array_equal(pred1, pred2)
        np.testing.assert_array_equal(pred2, pred3)


@unittest.skipUnless(TENSORFLOW_AVAILABLE, "TensorFlow not installed")
class TestLSTMModel(unittest.TestCase):
    """Test LSTM deep learning model"""
    
    @classmethod
    def setUpClass(cls):
        """Set up LSTM test data"""
        cls.df = get_sample_dataframe()
        
        # Preprocess
        cls.preprocessor = TextPreprocessor()
        cls.df = cls.preprocessor.preprocess_dataframe(cls.df)
        
        # Initialize LSTM
        cls.lstm = LSTMModel(max_words=1000, max_len=20, embedding_dim=50)
        
        # Prepare sequences
        cls.X, cls.y = cls.lstm.prepare_sequences(
            cls.df['processed_text'], 
            cls.df['sentiment']
        )
        
        # Split
        from sklearn.model_selection import train_test_split
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.X, cls.y, test_size=0.3, random_state=TEST_CONFIG['random_seed']
        )
    
    def test_prepare_sequences(self):
        """Test sequence preparation"""
        self.assertIsNotNone(self.lstm.tokenizer)
        self.assertIsNotNone(self.lstm.label_mapping)
        self.assertEqual(self.X.shape[1], 20)  # max_len
        self.assertEqual(self.y.shape[1], 3)   # 3 classes (positive, negative, neutral)
    
    def test_build_model(self):
        """Test model building"""
        model = self.lstm.build_model(num_classes=3, architecture='simple')
        
        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape, (None, 20))
        self.assertEqual(model.output_shape, (None, 3))
    
    def test_build_bidirectional_lstm(self):
        """Test bidirectional architecture"""
        model = self.lstm.build_model(num_classes=3, architecture='bidirectional')
        self.assertIsNotNone(model)
    
    def test_build_cnn_lstm(self):
        """Test CNN-LSTM architecture"""
        model = self.lstm.build_model(num_classes=3, architecture='cnn_lstm')
        self.assertIsNotNone(model)
    
    @skip_slow_tests()
    def test_train(self):
        """Test LSTM training"""
        self.lstm.build_model(num_classes=3, architecture='simple')
        
        history = self.lstm.train(
            self.X_train, self.y_train,
            self.X_test, self.y_test,
            epochs=2,  # Small for testing
            batch_size=4,
            use_callbacks=False
        )
        
        self.assertIsNotNone(history)
        self.assertIn('accuracy', history.history)
        self.assertIn('loss', history.history)
    
    @skip_slow_tests()
    def test_predict(self):
        """Test LSTM prediction"""
        self.lstm.build_model(num_classes=3, architecture='simple')
        self.lstm.train(
            self.X_train, self.y_train,
            self.X_test, self.y_test,
            epochs=2, batch_size=4, use_callbacks=False
        )
        
        test_texts = ["I love this!", "Terrible experience", "It's okay"]
        
        predictions, confidence = self.lstm.predict(test_texts)
        
        self.assertEqual(len(predictions), 3)
        self.assertEqual(len(confidence), 3)
        self.assertTrue(all(c in ['positive', 'negative', 'neutral'] for c in predictions))
        self.assertTrue(all(0 <= c <= 1 for c in confidence))
    
    @skip_slow_tests()
    def test_save_load(self):
        """Test saving and loading LSTM"""
        self.lstm.build_model(num_classes=3, architecture='simple')
        self.lstm.train(
            self.X_train, self.y_train,
            self.X_test, self.y_test,
            epochs=1, batch_size=4, use_callbacks=False
        )
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            model_path = os.path.join(temp_dir, 'test_lstm.h5')
            tokenizer_path = os.path.join(temp_dir, 'test_tokenizer.pkl')
            
            # Save
            self.lstm.save(model_path, tokenizer_path)
            self.assertTrue(os.path.exists(model_path))
            self.assertTrue(os.path.exists(tokenizer_path))
            
            # Load into new instance
            new_lstm = LSTMModel()
            new_lstm.load(model_path, tokenizer_path)
            
            self.assertIsNotNone(new_lstm.model)
            self.assertIsNotNone(new_lstm.tokenizer)
            
            # Test prediction
            pred, conf = new_lstm.predict(["Test text"])
            self.assertEqual(len(pred), 1)
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_evaluate(self):
        """Test LSTM evaluation"""
        self.lstm.build_model(num_classes=3, architecture='simple')
        self.lstm.train(
            self.X_train, self.y_train,
            self.X_test, self.y_test,
            epochs=1, batch_size=4, use_callbacks=False
        )
        
        metrics = self.lstm.evaluate(self.X_test, self.y_test)
        
        self.assertIn('accuracy', metrics)
        self.assertIn('loss', metrics)
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)


class TestEnsembleModel(unittest.TestCase):
    """Test Ensemble model"""
    
    def setUp(self):
        self.ensemble = EnsembleModel()
        
        # Create mock models
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import MultinomialNB
        
        # Simple test data
        self.X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        self.y = np.array(['pos', 'pos', 'neg', 'neg'])
        
        # Train simple models
        model1 = LogisticRegression()
        model1.fit(self.X, self.y)
        
        model2 = MultinomialNB()
        # Make data positive for NB
        X_pos = np.abs(self.X)
        model2.fit(X_pos, self.y)
        
        self.ensemble.add_model('lr', model1, weight=0.6)
        self.ensemble.add_model('nb', model2, weight=0.4)
    
    def test_add_model(self):
        """Test adding models to ensemble"""
        self.assertEqual(len(self.ensemble.models), 2)
        self.assertIn('lr', self.ensemble.models)
        self.assertIn('nb', self.ensemble.models)
        self.assertEqual(self.ensemble.weights['lr'], 0.6)
    
    def test_predict(self):
        """Test ensemble prediction"""
        # This is a simplified test - real ensemble would need proper feature extraction
        predictions = self.ensemble.predict(self.X)
        
        self.assertEqual(len(predictions), len(self.X))


class TestFeatureExtractor(unittest.TestCase):
    """Test feature extraction"""
    
    @classmethod
    def setUpClass(cls):
        cls.texts = [
            "I love this product!",
            "Terrible experience, hate it",
            "It's okay, nothing special"
        ]
    
    def test_simple_extractor(self):
        """Test simple TF-IDF extractor"""
        extractor = SimpleFeatureExtractor(method='tfidf', max_features=10)
        features = extractor.fit_transform(self.texts)
        
        self.assertEqual(features.shape[0], 3)
        self.assertLessEqual(features.shape[1], 10)
    
    def test_advanced_extractor(self):
        """Test advanced feature extractor"""
        extractor = AdvancedFeatureExtractor(
            methods=['tfidf', 'lexicon', 'statistical'],
            max_features=10
        )
        features = extractor.fit_transform(self.texts)
        
        self.assertEqual(features.shape[0], 3)
        self.assertGreater(features.shape[1], 10)  # Multiple feature types
    
    def test_sentiment_lexicon_features(self):
        """Test sentiment lexicon feature extraction"""
        extractor = AdvancedFeatureExtractor(methods=['lexicon'])
        features = extractor.extract_sentiment_lexicon_features(self.texts)
        
        self.assertEqual(features.shape[0], 3)
        self.assertEqual(features.shape[1], 10)  # 10 lexicon features
    
    def test_punctuation_features(self):
        """Test punctuation feature extraction"""
        extractor = AdvancedFeatureExtractor(methods=[])
        features = extractor.extract_punctuation_features([
            "Hello!!!",
            "What???",
            "Normal."
        ])
        
        self.assertEqual(features.shape[0], 3)
        self.assertEqual(features[0][0], 3)  # 3 exclamation marks
        self.assertEqual(features[1][1], 3)  # 3 question marks


class TestPreprocessing(unittest.TestCase):
    """Test text preprocessing"""
    
    @classmethod
    def setUpClass(cls):
        download_nltk_data()
        cls.preprocessor = TextPreprocessor()
    
    def test_clean_text(self):
        """Test text cleaning"""
        text = "Check this out! https://example.com @user #hashtag"
        cleaned = self.preprocessor.clean_text(text)
        
        self.assertNotIn('https', cleaned)
        self.assertNotIn('@user', cleaned)
        self.assertNotIn('#', cleaned)
    
    def test_preprocess(self):
        """Test full preprocessing pipeline"""
        text = "I absolutely LOVE this product!!! 😍"
        processed = self.preprocessor.preprocess(text)
        
        self.assertIsInstance(processed, str)
        self.assertNotIn('!!!', processed)  # Punctuation removed or normalized
    
    def test_preprocess_dataframe(self):
        """Test dataframe preprocessing"""
        df = get_sample_dataframe()
        processed_df = self.preprocessor.preprocess_dataframe(df)
        
        self.assertIn('processed_text', processed_df.columns)
        self.assertGreater(len(processed_df), 0)
    
    def test_sentiment_scores(self):
        """Test VADER sentiment scoring"""
        text = "I love this amazing product!"
        scores = self.preprocessor.get_sentiment_scores(text)
        
        self.assertIn('pos', scores)
        self.assertIn('neg', scores)
        self.assertIn('neu', scores)
        self.assertIn('compound', scores)
        self.assertGreater(scores['pos'], scores['neg'])
    
    def test_edge_cases(self):
        """Test edge case handling"""
        edge_cases = SAMPLE_TEXTS['edge_cases']
        
        for text in edge_cases:
            # Should not raise exception
            try:
                processed = self.preprocessor.preprocess(text)
                self.assertIsInstance(processed, str)
            except Exception as e:
                self.fail(f"Preprocessing failed for '{text}': {e}")


class TestIntegration(unittest.TestCase):
    """Integration tests for complete pipeline"""
    
    @classmethod
    def setUpClass(cls):
        download_nltk_data()
    
    def test_full_pipeline(self):
        """Test complete training and prediction pipeline"""
        # 1. Load data
        df = get_sample_dataframe()
        self.assertGreater(len(df), 0)
        
        # 2. Preprocess
        preprocessor = TextPreprocessor()
        df = preprocessor.preprocess_dataframe(df)
        self.assertIn('processed_text', df.columns)
        
        # 3. Extract features
        extractor = SimpleFeatureExtractor(max_features=50)
        X = extractor.fit_transform(df['processed_text'])
        y = df['sentiment'].values
        
        # 4. Train model
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        models = TraditionalModels()
        model, metrics = models.train_logistic_regression(
            X_train, y_train, X_test, y_test
        )
        
        # 5. Verify metrics
        self.assertGreater(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)
        
        # 6. Test prediction on new text
        new_text = "I love this!"
        processed = preprocessor.preprocess(new_text)
        features = extractor.transform([processed])
        prediction = model.predict(features)
        
        self.assertEqual(len(prediction), 1)
        self.assertIn(prediction[0], ['positive', 'negative', 'neutral'])
    
    @skip_slow_tests()
    @unittest.skipUnless(TENSORFLOW_AVAILABLE, "TensorFlow not installed")
    def test_lstm_pipeline(self):
        """Test LSTM complete pipeline"""
        df = get_sample_dataframe()
        
        preprocessor = TextPreprocessor()
        df = preprocessor.preprocess_dataframe(df)
        
        lstm = LSTMModel(max_words=500, max_len=10, embedding_dim=32)
        X, y = lstm.prepare_sequences(df['processed_text'], df['sentiment'])
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        lstm.build_model(num_classes=3, architecture='simple')
        lstm.train(X_train, y_train, X_test, y_test, epochs=2, batch_size=4, use_callbacks=False)
        
        predictions, confidence = lstm.predict(["Test text"])
        self.assertEqual(len(predictions), 1)


class TestPerformance(unittest.TestCase):
    """Performance and stress tests"""
    
    def test_large_batch_prediction(self):
        """Test prediction on large batch"""
        # Create large dataset
        texts = ["This is test text number " + str(i) for i in range(1000)]
        
        preprocessor = TextPreprocessor()
        processed = [preprocessor.preprocess(t) for t in texts[:100]]  # Sample
        
        extractor = SimpleFeatureExtractor(max_features=100)
        features = extractor.fit_transform(processed)
        
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        # Dummy training
        y = ['positive'] * 50 + ['negative'] * 50
        model.fit(features, y)
        
        # Time prediction
        start = time.time()
        predictions = model.predict(features)
        elapsed = time.time() - start
        
        self.assertLess(elapsed, 10)  # Should complete in 10 seconds
        self.assertEqual(len(predictions), 100)
    
    def test_memory_efficiency(self):
        """Test memory usage"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create moderately large dataset
        texts = ["Sample text " * 10 for _ in range(1000)]
        
        preprocessor = TextPreprocessor()
        processed = [preprocessor.preprocess(t) for t in texts]
        
        extractor = SimpleFeatureExtractor(max_features=1000)
        features = extractor.fit_transform(processed[:100])
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should use less than 500MB additional memory
        self.assertLess(memory_increase, 500)


def run_tests():
    """Run all tests with proper output"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestModelResults))
    suite.addTests(loader.loadTestsFromTestCase(TestTraditionalModels))
    
    if TENSORFLOW_AVAILABLE:
        suite.addTests(loader.loadTestsFromTestCase(TestLSTMModel))
    
    suite.addTests(loader.loadTestsFromTestCase(TestEnsembleModel))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run tests
    success = run_tests()
    sys.exit(0 if success else 1)