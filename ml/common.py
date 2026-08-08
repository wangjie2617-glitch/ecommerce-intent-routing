"""Shared data loading, threshold tuning and evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
)

from app.labels import LABELS

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

# TODO 将文本数据变成矩阵向量
def labels_to_matrix(records: list[dict[str, Any]]) -> np.ndarray:
    matrix = np.zeros((len(records), len(LABELS)), dtype=np.float32)
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    for row, record in enumerate(records):
        for label in record["labels"]:
            matrix[row, label_to_id[label]] = 1.0
    return matrix


def tune_thresholds(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for index, label in enumerate(LABELS):
        best_threshold, best_f1 = 0.5, -1.0
        for threshold in np.arange(0.20, 0.81, 0.05):
            prediction = (probabilities[:, index] >= threshold).astype(int)
            score = f1_score(y_true[:, index], prediction, zero_division=0)
            if score > best_f1:
                best_threshold, best_f1 = float(threshold), float(score)
        thresholds[label] = round(best_threshold, 2)
    return thresholds


def apply_thresholds(probabilities: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    values = np.asarray([thresholds[label] for label in LABELS], dtype=np.float32)
    predictions = (probabilities >= values).astype(int)
    empty_rows = np.where(predictions.sum(axis=1) == 0)[0]
    for row in empty_rows:
        predictions[row, int(np.argmax(probabilities[row]))] = 1
    return predictions


def compute_metrics(y_true: np.ndarray, probabilities: np.ndarray, thresholds: dict[str, float]) -> dict[str, Any]:
    y_pred = apply_thresholds(probabilities, thresholds)
    return {
        "micro_f1": round(float(f1_score(y_true, y_pred, average="micro", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "samples_f1": round(float(f1_score(y_true, y_pred, average="samples", zero_division=0)), 4),
        "micro_precision": round(float(precision_score(y_true, y_pred, average="micro", zero_division=0)), 4),
        "micro_recall": round(float(recall_score(y_true, y_pred, average="micro", zero_division=0)), 4),
        "exact_match": round(float(accuracy_score(y_true, y_pred)), 4),
        "hamming_loss": round(float(hamming_loss(y_true, y_pred)), 4),
        "per_label": classification_report(
            y_true,
            y_pred,
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }


def save_confusion_figure(y_true: np.ndarray, y_pred: np.ndarray, path: Path, title: str) -> None:
    matrices = multilabel_confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(3, 3, figsize=(12, 11))
    for index, (axis, matrix) in enumerate(zip(axes.flat, matrices, strict=True)):
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(LABELS[index])
        axis.set_xticks([0, 1], labels=["预测否", "预测是"])
        axis.set_yticks([0, 1], labels=["实际否", "实际是"])
        for row in range(2):
            for column in range(2):
                axis.text(column, row, int(matrix[row, column]), ha="center", va="center")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_evaluation(
    model_name: str,
    metrics: dict[str, Any],
    thresholds: dict[str, float],
    report_dir: Path,
    extra: dict[str, Any] | None = None,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {"model": model_name, "metrics": metrics, "thresholds": thresholds, "extra": extra or {}}
    slug = model_name.lower().replace(" ", "_").replace("/", "_")
    (report_dir / f"{slug}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        f"# {model_name} 评估结果",
        "",
        "> 指标来自本项目合成测试集，只用于技术方案对比，不代表生产环境效果。",
        "",
        "## 总体指标",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
    ]
    for key in ["micro_f1", "macro_f1", "samples_f1", "micro_precision", "micro_recall", "exact_match", "hamming_loss"]:
        lines.append(f"| {key} | {metrics[key]:.4f} |")
    lines.extend(["", "## 分类别结果", "", "| 类别 | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"])
    for label in LABELS:
        row = metrics["per_label"][label]
        lines.append(f"| {label} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1-score']:.4f} | {int(row['support'])} |")
    lines.extend(["", "## 分类阈值", "", "| 类别 | 阈值 |", "|---|---:|"])
    for label in LABELS:
        lines.append(f"| {label} | {thresholds[label]:.2f} |")
    if extra:
        lines.extend(["", "## 训练信息", "", "```json", json.dumps(extra, ensure_ascii=False, indent=2), "```"])
    (report_dir / f"{slug}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
