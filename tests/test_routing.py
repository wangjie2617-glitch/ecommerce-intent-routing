from app.labels import LABELS
from app.routing import build_route


def score_map(**overrides: float) -> dict[str, float]:
    scores = {label: 0.05 for label in LABELS}
    scores.update(overrides)
    return scores


def test_multi_intent_uses_highest_business_priority() -> None:
    intents, route = build_route(
        "物流一直没更新，我想退款",
        score_map(**{"物流异常": 0.92, "退款退货": 0.88}),
        {label: 0.5 for label in LABELS},
    )
    assert {intent["label"] for intent in intents} == {"物流异常", "退款退货"}
    assert route["department"] in {"物流客服", "售后客服"}
    assert route["priority"] == "medium"
    assert route["manual_review"] is False


def test_low_confidence_falls_back_to_manual_review() -> None:
    intents, route = build_route(
        "有个事情想问问",
        score_map(**{"其他问题": 0.42}),
        {label: 0.5 for label in LABELS},
    )
    assert intents[0]["label"] == "其他问题"
    assert route["manual_review"] is True


def test_high_risk_term_forces_escalation() -> None:
    intents, route = build_route(
        "商品有问题，再不处理我要投诉消协",
        score_map(**{"商品质量": 0.89}),
        {label: 0.5 for label in LABELS},
    )
    assert any(intent["label"] == "投诉升级" for intent in intents)
    assert route["department"] == "客诉专员"
    assert route["priority"] == "urgent"
    assert route["manual_review"] is True


def test_low_scoring_sensitive_label_does_not_hijack_route() -> None:
    intents, route = build_route(
        "付款成功为什么还是待支付",
        score_map(**{"支付问题": 0.74, "优惠活动": 0.45, "账号问题": 0.36}),
        {label: 0.3 for label in LABELS},
    )
    assert [intent["label"] for intent in intents] == ["支付问题"]
    assert route["department"] == "支付支持"
