"""Business labels and routing rules for the customer-service domain."""

from __future__ import annotations

LABELS = [
    "商品咨询",
    "退款退货",
    "物流异常",
    "商品质量",
    "支付问题",
    "优惠活动",
    "账号问题",
    "投诉升级",
    "其他问题",
]

ROUTING_RULES = {
    "商品咨询": {"department": "售前客服", "priority": "low", "sla_minutes": 30},
    "退款退货": {"department": "售后客服", "priority": "medium", "sla_minutes": 20},
    "物流异常": {"department": "物流客服", "priority": "medium", "sla_minutes": 20},
    "商品质量": {"department": "质量售后", "priority": "high", "sla_minutes": 10},
    "支付问题": {"department": "支付支持", "priority": "high", "sla_minutes": 10},
    "优惠活动": {"department": "营销客服", "priority": "low", "sla_minutes": 30},
    "账号问题": {"department": "账户安全", "priority": "high", "sla_minutes": 10},
    "投诉升级": {"department": "客诉专员", "priority": "urgent", "sla_minutes": 5},
    "其他问题": {"department": "综合客服", "priority": "medium", "sla_minutes": 30},
}

PRIORITY_RANK = {"low": 0, "medium": 1, "high": 2, "urgent": 3}

HIGH_RISK_TERMS = (
    "投诉消协",
    "市场监管",
    "报警",
    "律师",
    "曝光",
    "人身伤害",
    "起火",
    "爆炸",
    "欺诈",
    "盗刷",
)

# Keywords are not used as a standalone classifier. They only rescue an
# explicit intent when its model score is slightly below the relative margin.
INTENT_HINTS = {
    "商品咨询": ("支持", "参数", "规格", "尺寸", "颜色", "材质", "配件", "补货"),
    "退款退货": ("退款", "退货", "换货", "取消订单", "不想要"),
    "物流异常": ("快递", "物流", "包裹", "签收", "中转站", "没收到"),
    "商品质量": ("坏了", "破损", "划痕", "异味", "掉色", "故障", "残次", "少了"),
    "支付问题": ("支付", "付款", "扣款", "交易", "银行卡", "分期"),
    "优惠活动": ("优惠券", "满减", "折扣", "返现", "赠品", "活动价"),
    "账号问题": ("账号", "账户", "登录", "验证码", "密码", "实名认证", "安全风险"),
    "投诉升级": ("投诉", "负责人", "主管", "平台介入", "有关部门", "客诉"),
    "其他问题": ("发票", "人工客服", "收货备注", "订单记录", "隐私政策", "评价"),
}

SENSITIVE_LABEL_FLOORS = {"投诉升级": 0.58, "账号问题": 0.50}
