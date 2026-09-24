"""
NBA Player Career Duration Prediction — Gaussian Naive Bayes

Predicts whether an NBA player will have a career lasting five years or more
using rookie-season performance features.

Expected input:
    data/extracted_nba_players_data.csv

Target:
    target_5yrs

Model:
    Gaussian Naive Bayes
"""

from pathlib import Path

import pandas as pd
from sklearn import metrics, naive_bayes, model_selection
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "extracted_nba_players_data.csv"
FIGURE_PATH = ROOT / "figures" / "confusion_matrix.png"


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def prepare_data(data):
    y = data["target_5yrs"]
    X = data.drop("target_5yrs", axis=1)
    return X, y


def train_model(X, y):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    nb = naive_bayes.GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)

    return nb, X_train, X_test, y_train, y_test, y_pred


def evaluate_model(model, y_test, y_pred):
    results = {
        "accuracy": metrics.accuracy_score(y_test, y_pred),
        "precision": metrics.precision_score(y_test, y_pred),
        "recall": metrics.recall_score(y_test, y_pred),
        "f1": metrics.f1_score(y_test, y_pred),
    }

    print("Model Evaluation")
    print("-" * 30)
    for name, value in results.items():
        print(f"{name.capitalize():<10}: {value:.4f}")

    cm = metrics.confusion_matrix(y_test, y_pred)
    display = metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=model.classes_
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax)
    ax.set_title("Gaussian Naive Bayes — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=180)
    plt.close()

    return results, cm


def main():
    data = load_data()
    X, y = prepare_data(data)

    print(f"Dataset shape: {data.shape}")
    print(f"Training/test split: 75% / 25%")

    nb, X_train, X_test, y_train, y_test, y_pred = train_model(X, y)
    evaluate_model(nb, y_test, y_pred)


if __name__ == "__main__":
    main()
