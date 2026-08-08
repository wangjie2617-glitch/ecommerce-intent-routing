"""Fine-tune a lightweight Chinese BERT for multi-label intent classification."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from app.config import BERT_DIR, DATA_DIR, REPORTS_DIR
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

DEFAULT_MODEL = "uer/chinese_roberta_L-2_H-128"


class IntentDataset(Dataset):
    def __init__(self, records: list[dict[str, object]], tokenizer, max_length: int = 96) -> None:
        self.labels = torch.tensor(labels_to_matrix(records), dtype=torch.float32)
        self.encodings = tokenizer(
            [str(record["text"]) for record in records],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[index], dtype=torch.long) for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_probabilities(model, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels")
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs).logits
            batches.append(torch.sigmoid(logits).cpu().numpy())
            batch["labels"] = labels
    return np.concatenate(batches, axis=0)

# TODO  bert 训练的逻辑 重点掌握
def train_bert(
    model_name: str = DEFAULT_MODEL,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    seed: int = 20260721,
) -> dict[str, object]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_records = load_jsonl(DATA_DIR / "train.jsonl")
    validation_records = load_jsonl(DATA_DIR / "validation.jsonl")
    test_records = load_jsonl(DATA_DIR / "test.jsonl")
    y_train = labels_to_matrix(train_records)
    y_validation = labels_to_matrix(validation_records)
    y_test = labels_to_matrix(test_records)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    id2label = {index: label for index, label in enumerate(LABELS)}
    label2id = {label: index for index, label in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    ).to(device)

    train_loader = DataLoader(IntentDataset(train_records, tokenizer), batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(IntentDataset(validation_records, tokenizer), batch_size=batch_size * 2)
    test_loader = DataLoader(IntentDataset(test_records, tokenizer), batch_size=batch_size * 2)

    positives = y_train.sum(axis=0)
    negatives = len(y_train) - positives
    # Synthetic labels are already reasonably balanced. Square-root weighting
    # prevents the model from chasing recall at the cost of excessive false positives.
    pos_weight = torch.tensor(
        np.clip(np.sqrt(negatives / np.maximum(positives, 1)), 1.0, 3.0),
        dtype=torch.float32,
        device=device,
    )
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(int(total_steps * 0.1), 1),
        num_training_steps=total_steps,
    )

    history: list[dict[str, float | int]] = []
    best_macro_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_thresholds = {label: 0.5 for label in LABELS}
    training_started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs).logits
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())

        validation_probabilities = collect_probabilities(model, validation_loader, device)
        thresholds = tune_thresholds(y_validation, validation_probabilities)
        validation_metrics = compute_metrics(y_validation, validation_probabilities, thresholds)
        epoch_summary = {
            "epoch": epoch,
            "train_loss": round(total_loss / max(len(train_loader), 1), 5),
            "validation_macro_f1": validation_metrics["macro_f1"],
            "validation_micro_f1": validation_metrics["micro_f1"],
        }
        history.append(epoch_summary)
        print(json.dumps(epoch_summary, ensure_ascii=False))
        if float(validation_metrics["macro_f1"]) > best_macro_f1:
            best_macro_f1 = float(validation_metrics["macro_f1"])
            best_thresholds = thresholds
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})

    training_seconds = time.perf_counter() - training_started
    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state")
    model.load_state_dict(best_state)
    model.to(device)

    inference_started = time.perf_counter()
    test_probabilities = collect_probabilities(model, test_loader, device)
    inference_ms_per_item = (time.perf_counter() - inference_started) * 1000 / len(test_records)
    metrics = compute_metrics(y_test, test_probabilities, best_thresholds)

    BERT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(BERT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(BERT_DIR)
    (BERT_DIR / "thresholds.json").write_text(
        json.dumps(best_thresholds, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    training_info = {
        "base_model": model_name,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "training_records": len(train_records),
        "training_seconds": round(training_seconds, 3),
        "inference_ms_per_item": round(inference_ms_per_item, 3),
        "best_validation_macro_f1": round(best_macro_f1, 4),
        "history": history,
    }
    (BERT_DIR / "training_info.json").write_text(
        json.dumps(training_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_evaluation("Lightweight Chinese BERT", metrics, best_thresholds, REPORTS_DIR, training_info)
    y_pred = apply_thresholds(test_probabilities, best_thresholds)
    save_confusion_figure(
        y_test,
        y_pred,
        REPORTS_DIR / "figures" / "bert_multilabel_confusion.png",
        "轻量中文BERT：逐标签混淆矩阵",
    )
    return {"model": "bert", "metrics": metrics, "thresholds": best_thresholds, "training": training_info}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune lightweight Chinese BERT")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    result = train_bert(args.model_name, args.epochs, args.batch_size, args.learning_rate)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
