# Lightweight Chinese BERT 评估结果

> 指标来自本项目合成测试集，只用于技术方案对比，不代表生产环境效果。

## 总体指标

| 指标 | 数值 |
|---|---:|
| micro_f1 | 0.6588 |
| macro_f1 | 0.6685 |
| samples_f1 | 0.6544 |
| micro_precision | 0.5435 |
| micro_recall | 0.8360 |
| exact_match | 0.2733 |
| hamming_loss | 0.1389 |

## 分类别结果

| 类别 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 商品咨询 | 0.6620 | 0.9592 | 0.7833 | 49 |
| 退款退货 | 0.8889 | 0.6780 | 0.7692 | 59 |
| 物流异常 | 0.3152 | 0.7073 | 0.4361 | 41 |
| 商品质量 | 0.5053 | 1.0000 | 0.6713 | 48 |
| 支付问题 | 0.6129 | 0.6786 | 0.6441 | 56 |
| 优惠活动 | 0.5775 | 0.8542 | 0.6891 | 48 |
| 账号问题 | 0.3086 | 0.9615 | 0.4673 | 26 |
| 投诉升级 | 0.5413 | 0.8806 | 0.6705 | 67 |
| 其他问题 | 0.8750 | 0.8974 | 0.8861 | 39 |

## 分类阈值

| 类别 | 阈值 |
|---|---:|
| 商品咨询 | 0.45 |
| 退款退货 | 0.55 |
| 物流异常 | 0.40 |
| 商品质量 | 0.45 |
| 支付问题 | 0.35 |
| 优惠活动 | 0.40 |
| 账号问题 | 0.30 |
| 投诉升级 | 0.50 |
| 其他问题 | 0.55 |

## 训练信息

```json
{
  "base_model": "uer/chinese_roberta_L-2_H-128",
  "device": "cpu",
  "gpu": null,
  "epochs": 10,
  "batch_size": 32,
  "learning_rate": 0.0001,
  "training_records": 1800,
  "training_seconds": 73.729,
  "inference_ms_per_item": 0.677,
  "best_validation_macro_f1": 0.641,
  "history": [
    {
      "epoch": 1,
      "train_loss": 0.81158,
      "validation_macro_f1": 0.266,
      "validation_micro_f1": 0.2676
    },
    {
      "epoch": 2,
      "train_loss": 0.73504,
      "validation_macro_f1": 0.406,
      "validation_micro_f1": 0.4182
    },
    {
      "epoch": 3,
      "train_loss": 0.65614,
      "validation_macro_f1": 0.5151,
      "validation_micro_f1": 0.4921
    },
    {
      "epoch": 4,
      "train_loss": 0.57746,
      "validation_macro_f1": 0.5345,
      "validation_micro_f1": 0.5084
    },
    {
      "epoch": 5,
      "train_loss": 0.52336,
      "validation_macro_f1": 0.531,
      "validation_micro_f1": 0.5124
    },
    {
      "epoch": 6,
      "train_loss": 0.48461,
      "validation_macro_f1": 0.5964,
      "validation_micro_f1": 0.5747
    },
    {
      "epoch": 7,
      "train_loss": 0.45675,
      "validation_macro_f1": 0.6032,
      "validation_micro_f1": 0.5938
    },
    {
      "epoch": 8,
      "train_loss": 0.43832,
      "validation_macro_f1": 0.6142,
      "validation_micro_f1": 0.603
    },
    {
      "epoch": 9,
      "train_loss": 0.42503,
      "validation_macro_f1": 0.641,
      "validation_micro_f1": 0.6268
    },
    {
      "epoch": 10,
      "train_loss": 0.41917,
      "validation_macro_f1": 0.6242,
      "validation_micro_f1": 0.615
    }
  ]
}
```
