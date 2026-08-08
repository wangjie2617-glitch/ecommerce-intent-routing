"""Generate data and train both models in a reproducible sequence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import REPORTS_DIR  # noqa: E402
from ml.data_generator import build_dataset  # noqa: E402
from ml.train_baseline import train_baseline  # noqa: E402
from ml.train_bert import train_bert  # noqa: E402


def write_comparison(baseline: dict[str, object], bert: dict[str, object]) -> None:
    baseline_metrics = baseline["metrics"]
    bert_metrics = bert["metrics"]
    lines = [
        "# 模型对比与选择",
        "",
        "> 结果来自可复现的合成测试集，只用于展示技术流程，不能代表真实生产效果。",
        "",
        "| 模型 | Micro-F1 | Macro-F1 | Samples-F1 | Exact Match | Hamming Loss |",
        "|---|---:|---:|---:|---:|---:|",
        f"| TF-IDF + Logistic Regression | {baseline_metrics['micro_f1']:.4f} | {baseline_metrics['macro_f1']:.4f} | {baseline_metrics['samples_f1']:.4f} | {baseline_metrics['exact_match']:.4f} | {baseline_metrics['hamming_loss']:.4f} |",
        f"| 轻量中文BERT | {bert_metrics['micro_f1']:.4f} | {bert_metrics['macro_f1']:.4f} | {bert_metrics['samples_f1']:.4f} | {bert_metrics['exact_match']:.4f} | {bert_metrics['hamming_loss']:.4f} |",
        "",
        "## 选择策略",
        "",
        "API默认优先加载BERT；如果BERT产物不存在，则自动回退到TF-IDF基线。生产环境应在获得脱敏真实数据后重新评估，不应只根据当前合成集指标决定模型。",
    ]
    (REPORTS_DIR / "model_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print("[1/3] 生成可复现合成数据")
    summary = build_dataset()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("[2/3] 训练TF-IDF基线")
    baseline = train_baseline()
    print("[3/3] 微调轻量中文BERT")
    bert = train_bert()
    write_comparison(baseline, bert)
    print("训练完成，评估结果位于 reports/。")


if __name__ == "__main__":
    main()

