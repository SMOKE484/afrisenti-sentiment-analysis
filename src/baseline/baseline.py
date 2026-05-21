import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


LANGUAGES = {
    "hau": "Hausa",
    "yor": "Yoruba",
    "ibo": "Igbo"
}

LABEL_NAMES = ["negative", "neutral", "positive"]


def load_processed_data(lang_code):
    """
    Loads the cleaned CSVs from data/processed/ for a given language.
    Returns train, validation, and test DataFrames.
    """
    splits = {}
    for split in ["train", "validation", "test"]:
        path = f"data/processed/{lang_code}_{split}_clean.csv"
        if not os.path.exists(path):
            print(f"File not found: {path}. Run preprocess.py first.")
            return None
        splits[split] = pd.read_csv(path)
    return splits


def train_baseline(lang_code):
    """
    Trains a TF-IDF + Logistic Regression baseline classifier for a
    given language. This serves as our lower-bound benchmark before
    we move to transformer fine-tuning.

    We use character n-grams (analyzer='char_wb') instead of word n-grams
    because African languages are morphologically rich - one root word
    can have many forms. Character n-grams handle this better than
    splitting by spaces.
    """
    print(f"\n{'='*55}")
    print(f"Training baseline for {LANGUAGES[lang_code]} ({lang_code})")
    print("="*55)

    splits = load_processed_data(lang_code)
    if splits is None:
        return None

    train_df = splits["train"]
    val_df = splits["validation"]
    test_df = splits["test"]

    X_train = train_df["cleaned_text"].fillna("")
    y_train = train_df["label"]

    X_val = val_df["cleaned_text"].fillna("")
    y_val = val_df["label"]

    X_test = test_df["cleaned_text"].fillna("")
    y_test = test_df["label"]

    # TF-IDF with character n-grams (2 to 5 characters)
    print("\nFitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        max_features=50000,
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    # Logistic Regression with class_weight='balanced' to handle imbalance
    print("Training Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=1.0,
        solver="lbfgs"
    )
    model.fit(X_train_tfidf, y_train)

    # Evaluate on validation set
    val_preds = model.predict(X_val_tfidf)
    val_f1 = f1_score(y_val, val_preds, average="weighted")
    val_acc = accuracy_score(y_val, val_preds)

    # Evaluate on test set
    test_preds = model.predict(X_test_tfidf)
    test_f1 = f1_score(y_test, test_preds, average="weighted")
    test_acc = accuracy_score(y_test, test_preds)

    print(f"\nValidation  -> Weighted F1: {val_f1:.4f} | Accuracy: {val_acc:.4f}")
    print(f"Test        -> Weighted F1: {test_f1:.4f} | Accuracy: {test_acc:.4f}")

    print("\nDetailed Test Classification Report:")
    print(classification_report(
        y_test,
        test_preds,
        target_names=LABEL_NAMES,
        digits=4
    ))

    return {
        "lang_code": lang_code,
        "model": model,
        "vectorizer": vectorizer,
        "val_f1": val_f1,
        "test_f1": test_f1,
        "test_acc": test_acc,
        "test_preds": test_preds,
        "y_test": y_test
    }


def plot_confusion_matrix(results, lang_code):
    """
    Plots and saves a confusion matrix for a given language.
    This will be reused later in the error analysis section.
    """
    os.makedirs("results", exist_ok=True)

    cm = confusion_matrix(results["y_test"], results["test_preds"])

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES
    )
    plt.title(
        f"Baseline Confusion Matrix - {LANGUAGES[lang_code]} ({lang_code})"
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    save_path = f"results/{lang_code}_baseline_confusion_matrix.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved: {save_path}")


def save_model(results, lang_code):
    """
    Saves the trained model and vectorizer so we do not have to retrain.
    """
    os.makedirs("results/models", exist_ok=True)

    joblib.dump(
        results["model"],
        f"results/models/{lang_code}_baseline_model.pkl"
    )
    joblib.dump(
        results["vectorizer"],
        f"results/models/{lang_code}_baseline_vectorizer.pkl"
    )
    print(f"Model saved: results/models/{lang_code}_baseline_model.pkl")


def summarise_all_results(all_results):
    """
    Prints a summary table comparing baseline performance
    across all three languages.
    """
    print("\n" + "="*55)
    print("BASELINE RESULTS SUMMARY")
    print("="*55)
    print(f"{'Language':<12} {'Val F1':>10} {'Test F1':>10} {'Test Acc':>10}")
    print("-"*45)

    for lang_code, results in all_results.items():
        print(
            f"{LANGUAGES[lang_code]:<12}"
            f"{results['val_f1']:>10.4f}"
            f"{results['test_f1']:>10.4f}"
            f"{results['test_acc']:>10.4f}"
        )


if __name__ == "__main__":
    all_results = {}

    for lang_code in LANGUAGES:
        results = train_baseline(lang_code)
        if results:
            plot_confusion_matrix(results, lang_code)
            save_model(results, lang_code)
            all_results[lang_code] = results

    summarise_all_results(all_results)