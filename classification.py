"""Supervised Arabic document classification: char n-gram TF-IDF + logistic regression."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .arabic import normalize_arabic

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DATA = PROJECT_ROOT / "data" / "training" / "labels.csv"


@dataclass(frozen=True)
class Prediction:
    category: str
    score: float  # uncalibrated model probability, not a reliability guarantee


def read_labeled_csv(path: Path) -> tuple[list[str], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        records = list(csv.DictReader(file))
    if not records or any(not row.get("text") or not row.get("category") for row in records):
        raise ValueError(f"Invalid or empty training/evaluation dataset: {path}")
    return [row["text"] for row in records], [row["category"] for row in records]


def train_classifier(path: Path = TRAIN_DATA) -> Pipeline:
    texts, labels = read_labeled_csv(path)
    if len(set(labels)) < 2:
        raise ValueError("At least two classes are required")
    classifier = Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    preprocessor=normalize_arabic,
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    sublinear_tf=True,
                    max_features=25_000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1_000, random_state=42, class_weight="balanced"),
            ),
        ]
    )
    classifier.fit(texts, labels)
    return classifier


def predict_category(classifier: Pipeline, document_text: str) -> Prediction:
    if not document_text.strip():
        raise ValueError("Cannot classify an empty document")
    probabilities = classifier.predict_proba([document_text[:12_000]])[0]
    index = int(probabilities.argmax())
    label = str(classifier.classes_[index])
    return Prediction(category=label, score=float(probabilities[index]))
