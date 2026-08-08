# 模型对比与选择

> 指标来自2400条可复现合成数据中的独立测试集，只用于技术方案对比，不代表真实生产效果。

| 模型 | Micro-F1 | Macro-F1 | Samples-F1 | Exact Match | Hamming Loss | 批量推理耗时/条 |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.5971 | 0.5890 | 0.5691 | 0.3800 | 0.1344 | 0.060 ms |
| 轻量中文BERT | **0.6588** | **0.6685** | **0.6544** | 0.2733 | 0.1389 | 0.677 ms |

## 结论

- BERT的Macro-F1比基线高0.0795，即7.95个百分点，因此API默认优先加载BERT；
- TF-IDF的Exact Match和Hamming Loss更好，且推理速度更快，说明简单模型仍然是有价值的业务基线；
- BERT在“物流异常”和“账号问题”上仍有较多误报，因此在线路由增加了绝对阈值、相对分差、显式关键词和敏感类别下限；
- BERT不可用时，服务自动回退到TF-IDF模型；
- 正式上线前必须使用脱敏真实数据重新训练、调阈值并开展人工验收。

详细结果见：

- [TF-IDF评估](tfidf_baseline.md)
- [BERT评估](lightweight_chinese_bert.md)
- [数据声明](../docs/data_statement.md)

