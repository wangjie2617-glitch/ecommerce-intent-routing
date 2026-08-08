"""Convert model scores into explainable business routing decisions."""

from __future__ import annotations

from collections.abc import Mapping

from app.labels import (
    HIGH_RISK_TERMS,
    INTENT_HINTS,
    PRIORITY_RANK,
    ROUTING_RULES,
    SENSITIVE_LABEL_FLOORS,
)


#  TODO
def build_route(
    text: str,
    scores: Mapping[str, float],
    thresholds: Mapping[str, float],
    manual_review_threshold: float = 0.55,
) -> tuple[list[dict[str, float | str]], dict[str, object]]:
    """Select intents and route a request with transparent fallback rules."""

    best_label, best_score = max(scores.items(), key=lambda item: item[1])
    selected = []
    for label, score in scores.items():
        absolute_threshold = max(
            thresholds.get(label, 0.5),
            SENSITIVE_LABEL_FLOORS.get(label, 0.45),
        )
        has_explicit_hint = any(term in text for term in INTENT_HINTS[label])
        close_to_best = score >= best_score - 0.10
        if score >= absolute_threshold and (close_to_best or has_explicit_hint):
            selected.append({"label": label, "score": round(float(score), 4)})
    selected.sort(key=lambda item: float(item["score"]), reverse=True)

    # "其他问题" is a fallback class. When it is the strongest result and no
    # competing label has explicit evidence, avoid manufacturing multi-intents.
    if best_label == "其他问题" and selected:
        explicit_competitors = [
            item
            for item in selected
            if item["label"] != "其他问题"
            and any(term in text for term in INTENT_HINTS[str(item["label"])])
        ]
        selected = [item for item in selected if item["label"] == "其他问题"] + explicit_competitors
    elif len(selected) > 1:
        selected = [item for item in selected if item["label"] != "其他问题"]

    selected = selected[:3]

    reasons: list[str] = []
    if not selected:
        selected = [{"label": best_label, "score": round(float(best_score), 4)}]
        reasons.append("没有类别达到独立阈值，保留最高分结果并转人工复核")

    selected_labels = [str(item["label"]) for item in selected]
    rules = [ROUTING_RULES[label] for label in selected_labels]
    primary_rule = max(rules, key=lambda rule: PRIORITY_RANK[str(rule["priority"])])

    max_score = max(float(item["score"]) for item in selected)
    manual_review = max_score < manual_review_threshold
    if manual_review:
        reasons.append(f"最高置信度 {max_score:.2f} 低于人工复核阈值 {manual_review_threshold:.2f}")

    matched_risk_terms = [term for term in HIGH_RISK_TERMS if term in text]
    if matched_risk_terms:
        manual_review = True
        primary_rule = ROUTING_RULES["投诉升级"]
        reasons.append(f"命中高风险词：{', '.join(matched_risk_terms)}")
        if "投诉升级" not in selected_labels:
            selected.append({"label": "投诉升级", "score": 1.0})

    if len(selected) > 1:
        reasons.append(f"识别到多重诉求：{', '.join(str(item['label']) for item in selected)}")
    if not reasons:
        reasons.append("模型置信度达到阈值，按最高业务优先级路由")

    route = {
        "department": primary_rule["department"],
        "priority": primary_rule["priority"],
        "sla_minutes": primary_rule["sla_minutes"],
        "manual_review": manual_review,
        "reasons": reasons,
    }
    return selected, route
