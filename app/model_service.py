"""Lazy-loading inference service supporting both baseline and BERT artifacts."""

from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Lock

import joblib
import numpy as np

from app.config import BASELINE_DIR, BERT_DIR, MANUAL_REVIEW_THRESHOLD, MODEL_TYPE
from app.labels import LABELS
from app.routing import build_route


class ModelNotReadyError(RuntimeError):
    """Raised when no trained artifact is available."""

# TODO service
class IntentModelService:
    def __init__(self, preferred_model: str = MODEL_TYPE) -> None:
        self.preferred_model = preferred_model
        self.model_type: str | None = None
        self.model = None
        self.tokenizer = None
        self.thresholds = {label: 0.5 for label in LABELS}
        self.device = "cpu"
        self._lock = Lock()

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        with self._lock:
            if self.is_ready:
                return
            candidates = (
                [self.preferred_model]
                if self.preferred_model in {"baseline", "bert"}
                else ["bert", "baseline"]
            )
            errors: list[str] = []
            for candidate in candidates:
                try:
                    if candidate == "bert" and (BERT_DIR / "config.json").exists():
                        self._load_bert(BERT_DIR)
                        return
                    if candidate == "baseline" and (BASELINE_DIR / "model.joblib").exists():
                        self._load_baseline(BASELINE_DIR)
                        return
                except Exception as exc:  # pragma: no cover - defensive startup path
                    errors.append(f"{candidate}: {exc}")
            detail = "; ".join(errors) if errors else "未找到训练产物"
            raise ModelNotReadyError(
                f"模型尚未准备好（{detail}）。请先运行 python scripts/train_all.py。"
            )

    def _load_baseline(self, artifact_dir: Path) -> None:
        bundle = joblib.load(artifact_dir / "model.joblib")
        self.model = bundle["pipeline"]
        self.thresholds = bundle["thresholds"]
        self.model_type = "tfidf-logistic-regression"

    def _load_bert(self, artifact_dir: Path) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(artifact_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(artifact_dir)
        self.model.to(self.device)
        self.model.eval()
        threshold_path = artifact_dir / "thresholds.json"
        if threshold_path.exists():
            self.thresholds = json.loads(threshold_path.read_text(encoding="utf-8"))
        self.model_type = "lightweight-chinese-bert"

    def _predict_scores(self, texts: list[str]) -> np.ndarray:
        if not self.is_ready:
            self.load()
        if self.model_type == "tfidf-logistic-regression":
            return np.asarray(self.model.predict_proba(texts), dtype=float)

        import torch

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = self.model(**encoded).logits
            return torch.sigmoid(logits).cpu().numpy()

    def predict(self, texts: list[str]) -> list[dict[str, object]]:
        started = time.perf_counter()
        probabilities = self._predict_scores(texts)
        elapsed_ms = (time.perf_counter() - started) * 1000
        per_item_ms = elapsed_ms / max(len(texts), 1)

        results: list[dict[str, object]] = []
        for text, row in zip(texts, probabilities, strict=True):
            scores = {label: float(row[index]) for index, label in enumerate(LABELS)}
            intents, route = build_route(
                text,
                scores,
                self.thresholds,
                manual_review_threshold=MANUAL_REVIEW_THRESHOLD,
            )
            results.append(
                {
                    "text": text,
                    "intents": intents,
                    "route": route,
                    "model_type": self.model_type,
                    "latency_ms": round(per_item_ms, 2),
                }
            )
        return results


model_service = IntentModelService()

