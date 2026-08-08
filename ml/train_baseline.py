"""Train the TF-IDF + one-vs-rest logistic regression baseline."""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from app.config import BASELINE_DIR, DATA_DIR, REPORTS_DIR
from app.labels import LABELS
from ml.common import (
    apply_thresholds,
    compute_metrics,
    labels_to_matrix,
    load_jsonl,
    save_confusion_figure,
    save_evaluation,
    tune_thresholds,
)


# TODO 深度理解
def train_baseline() -> dict[str, object]:
    train = load_jsonl(DATA_DIR / "train.jsonl")
    validation = load_jsonl(DATA_DIR / "validation.jsonl")
    test = load_jsonl(DATA_DIR / "test.jsonl")
    y_train = labels_to_matrix(train)
    y_validation = labels_to_matrix(validation)
    y_test = labels_to_matrix(test)

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 5),
                    min_df=2,
                    max_features=50_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                OneVsRestClassifier(
                    LogisticRegression(max_iter=1200, class_weight="balanced", solver="liblinear")
                ),
            ),
        ]
    )
    started = time.perf_counter()
    pipeline.fit([record["text"] for record in train], y_train)
    training_seconds = time.perf_counter() - started

    validation_probabilities = np.asarray(
        pipeline.predict_proba([record["text"] for record in validation]), dtype=float
    )
    thresholds = tune_thresholds(y_validation, validation_probabilities)

    inference_started = time.perf_counter()
    test_probabilities = np.asarray(
        pipeline.predict_proba([record["text"] for record in test]), dtype=float
    )
    inference_ms_per_item = (time.perf_counter() - inference_started) * 1000 / len(test)
    metrics = compute_metrics(y_test, test_probabilities, thresholds)

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": pipeline, "thresholds": thresholds, "labels": LABELS},
        BASELINE_DIR / "model.joblib",
    )
    (BASELINE_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "model": "TF-IDF + OneVsRest LogisticRegression",
                "labels": LABELS,
                "training_records": len(train),
                "training_seconds": round(training_seconds, 3),
                "inference_ms_per_item": round(inference_ms_per_item, 3),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    save_evaluation(
        "TFIDF Baseline",
        metrics,
        thresholds,
        REPORTS_DIR,
        {
            "training_records": len(train),
            "training_seconds": round(training_seconds, 3),
            "inference_ms_per_item": round(inference_ms_per_item, 3),
        },
    )
    y_pred = apply_thresholds(test_probabilities, thresholds)
    save_confusion_figure(
        y_test,
        y_pred,
        REPORTS_DIR / "figures" / "baseline_multilabel_confusion.png",
        "TF-IDF基线：逐标签混淆矩阵",
    )
    return {"model": "baseline", "metrics": metrics, "thresholds": thresholds}


if __name__ == "__main__":
    result = train_baseline()
    print(json.dumps(result, ensure_ascii=False, indent=2))

