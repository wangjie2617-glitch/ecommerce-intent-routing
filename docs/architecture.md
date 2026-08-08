# 系统架构

```mermaid
flowchart LR
    U["客服消息 / 批量文件"] --> API["FastAPI接口"]
    API --> MS["模型服务"]
    MS --> BERT["轻量中文BERT"]
    MS -. "产物缺失时回退" .-> BASE["TF-IDF + LR"]
    BERT --> SCORE["9类概率"]
    BASE --> SCORE
    SCORE --> DEC["阈值与业务决策层"]
    DEC --> A["绝对分类阈值"]
    DEC --> B["相对分差"]
    DEC --> C["显式意图词"]
    DEC --> D["高风险词"]
    A --> ROUTE["工单路由"]
    B --> ROUTE
    C --> ROUTE
    D --> ROUTE
    ROUTE --> OUT["意图 / 部门 / 优先级 / SLA / 人工复核"]
```

## 离线训练链路

```mermaid
flowchart LR
    GEN["可复现模板生成器"] --> SPLIT["Train / Validation / Test"]
    SPLIT --> TF["TF-IDF基线训练"]
    SPLIT --> BT["BERT微调"]
    TF --> TUNE["逐标签阈值调优"]
    BT --> TUNE
    TUNE --> EVAL["多标签评估"]
    EVAL --> REPORT["Markdown / JSON / 混淆矩阵"]
    EVAL --> ART["模型与阈值产物"]
```

## 核心设计

1. **模型和规则分层**：模型只输出概率，部门、SLA和人工复核由可解释业务规则决定。
2. **双模型回退**：优先加载BERT，产物不存在时自动使用TF-IDF模型。
3. **多标签阈值**：每个类别在验证集上单独调节阈值，避免统一0.5造成类别偏差。
4. **敏感类别保护**：账号与投诉类别设置更高绝对阈值；高风险词强制转人工。
5. **通用接口**：单条与批量API使用一致响应结构，方便后续RAG Agent调用。

