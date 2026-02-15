import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent))

from sentiment_analysis.config import MODELS_DIR, get_config
from src.data_loader import DataLoader
from src.feature_extractor import FeatureExtractor
from src.models import LSTMModel, TraditionalModels
from src.preprocessor import TextPreprocessor, download_nltk_data


def load_dataset(dataset: str, data_path: str = None):
    loader = DataLoader()
    if dataset == "twitter":
        df = loader.load_twitter_data(data_path)
        text_col = "text"
    elif dataset == "imdb":
        df = loader.load_imdb_data(data_path)
        text_col = "review"
    elif dataset == "amazon":
        df = loader.load_amazon_data(data_path)
        text_col = "review_text"
    else:
        raise ValueError("dataset must be one of: twitter, imdb, amazon")

    if "sentiment" not in df.columns:
        raise ValueError("Loaded dataset does not contain a sentiment column")

    df = df[[text_col, "sentiment"]].dropna().copy()
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    df = df[df["sentiment"].isin(["positive", "neutral", "negative"])].copy()

    if df.empty:
        raise ValueError("No valid rows after sentiment normalization")

    return df, text_col


def train_pipeline(dataset: str, data_path: str, feature_method: str, enable_lstm: bool):
    cfg = get_config()
    MODELS_DIR.mkdir(exist_ok=True)

    print("Downloading NLTK resources (if needed)...")
    download_nltk_data()

    print(f"Loading dataset={dataset} path={data_path or 'default sample'}")
    df, text_col = load_dataset(dataset, data_path)
    print(f"Rows loaded: {len(df)}")
    print(df["sentiment"].value_counts().to_string())

    preprocessor = TextPreprocessor(
        remove_stopwords=cfg["preprocessing"]["remove_stopwords"],
        lemmatize=cfg["preprocessing"]["lemmatization"],
        stem=cfg["preprocessing"]["stemming"],
    )

    df = preprocessor.preprocess_dataframe(df, text_column=text_col)
    df = df[df["processed_text"].str.len() > 0].copy()

    extractor = FeatureExtractor(
        method=feature_method,
        max_features=cfg["feature"]["tfidf"]["max_features"],
    )

    X = extractor.fit_transform(df["processed_text"].tolist())
    y = df["sentiment"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
        stratify=y,
    )

    traditional = TraditionalModels()
    timings = {}

    if feature_method != "word2vec":
        t0 = time.time()
        traditional.train_naive_bayes(X_train, y_train, X_test, y_test)
        timings["naive_bayes"] = time.time() - t0
    else:
        print("Skipping Naive Bayes for word2vec features (MultinomialNB expects count-like features).")

    t0 = time.time()
    traditional.train_logistic_regression(X_train, y_train, X_test, y_test)
    timings["logistic_regression"] = time.time() - t0

    t0 = time.time()
    traditional.train_svm(X_train, y_train, X_test, y_test)
    timings["svm"] = time.time() - t0

    best_name, best_model = traditional.get_best_model()

    # Save artifacts
    with open(MODELS_DIR / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    extractor.save(str(MODELS_DIR / "feature_extractor.pkl"))

    for model_name in traditional.models:
        traditional.save_model(model_name, str(MODELS_DIR / f"{model_name}.pkl"))

    with open(MODELS_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    lstm_metrics = None
    if enable_lstm:
        try:
            t0 = time.time()
            lstm = LSTMModel(
                max_words=cfg["lstm"]["max_words"],
                max_len=cfg["lstm"]["max_len"],
                embedding_dim=cfg["lstm"]["embedding_dim"],
            )
            X_seq, y_seq = lstm.prepare_sequences(df["processed_text"].tolist(), y)
            X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
                X_seq,
                y_seq,
                test_size=cfg["data"]["test_size"],
                random_state=cfg["data"]["random_state"],
            )
            lstm.build_model(num_classes=y_seq.shape[1])
            lstm.train(
                X_train_l,
                y_train_l,
                X_test_l,
                y_test_l,
                epochs=cfg["lstm"]["epochs"],
                batch_size=cfg["lstm"]["batch_size"],
            )
            lstm.save(
                str(MODELS_DIR / "lstm_model.h5"),
                str(MODELS_DIR / "lstm_tokenizer.pkl"),
            )
            timings["lstm"] = time.time() - t0
            lstm_metrics = lstm.evaluate(X_test_l, y_test_l)
        except Exception as ex:
            print(f"LSTM training skipped/failed: {ex}")

    report = {
        "dataset": dataset,
        "data_path": data_path,
        "feature_method": feature_method,
        "samples": len(df),
        "best_model": best_name,
        "timings_seconds": timings,
        "lstm_metrics": lstm_metrics,
    }

    with open(MODELS_DIR / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nTraining complete")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Train sentiment models")
    parser.add_argument("--dataset", choices=["twitter", "imdb", "amazon"], default="twitter")
    parser.add_argument("--data-path", default=None, help="Path to CSV/ZIP/XLSX file")
    parser.add_argument("--feature-method", choices=["tfidf", "count", "word2vec"], default="tfidf")
    parser.add_argument("--no-lstm", action="store_true")
    args = parser.parse_args()

    train_pipeline(
        dataset=args.dataset,
        data_path=args.data_path,
        feature_method=args.feature_method,
        enable_lstm=not args.no_lstm,
    )


if __name__ == "__main__":
    main()
