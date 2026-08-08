"""Generate a reproducible synthetic multi-label e-commerce intent dataset.

The generator uses separate wording pools for train, validation and test to
reduce exact-template leakage. It is a demonstration dataset, not production
customer data. See docs/data_statement.md for the limitations.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from app.config import DATA_DIR
from app.labels import LABELS

PRODUCTS = ["蓝牙耳机", "机械键盘", "保温杯", "运动鞋", "充电宝", "羽绒服", "咖啡机", "护肤套装"]
COLORS = ["黑色", "白色", "蓝色", "灰色", "米色"]
PREFIXES = ["你好，", "客服您好，", "麻烦看一下，", "想咨询一下，", "请问，", ""]
SUFFIXES = ["谢谢", "麻烦尽快处理", "在线等", "请回复一下", "", "可以帮忙看看吗"]
CONNECTORS = ["，另外", "，而且", "；同时", "，还有", "。顺便问下，"]

TRAIN_TEMPLATES = {
    "商品咨询": [
        "{product}支持哪些功能", "{product}有没有{color}款", "这款{product}适合日常使用吗",
        "{product}的尺寸和材质是什么", "现在下单{product}包含哪些配件", "{product}能不能全国联保",
        "想了解一下{product}的规格", "{product}什么时候补货",
    ],
    "退款退货": [
        "这个订单我不想要了申请退款", "收到后不合适想退货", "请帮我取消订单并原路退款",
        "七天无理由退货怎么操作", "退货地址发我一下", "退款一直没有到账",
        "我买错型号了需要换货", "订单还没发出请直接退款",
    ],
    "物流异常": [
        "快递三天没有更新了", "物流显示签收但我没收到", "包裹一直停在中转站",
        "快递送错地址了", "预计昨天到现在还没到", "物流单号查不到信息",
        "包裹被退回去了", "快递员一直没有联系我",
    ],
    "商品质量": [
        "{product}刚用就坏了", "收到的{product}有明显划痕", "商品破损而且无法正常使用",
        "实物和页面描述不一致", "{product}有异味并且掉色", "包装完好但里面配件缺失",
        "使用两次就出现故障", "收到的是残次品",
    ],
    "支付问题": [
        "已经扣款但订单还是待支付", "付款时一直提示交易失败", "重复扣了两次钱",
        "微信支付成功订单却关闭了", "银行卡支付被拒绝", "支付页面打不开",
        "分期付款为什么用不了", "订单金额和付款金额不一致",
    ],
    "优惠活动": [
        "优惠券为什么不能使用", "满减活动怎么参加", "新人券在哪里领取",
        "直播间价格和结算价不一样", "会员折扣可以叠加吗", "活动赠品什么时候发",
        "优惠券显示已经失效", "参加活动后为什么没有返现",
    ],
    "账号问题": [
        "账号突然登录不上去了", "收不到登录验证码", "手机号码怎么更换",
        "账号提示存在安全风险", "忘记密码怎么找回", "我的账号好像被别人登录了",
        "实名认证一直失败", "账号被限制下单了",
    ],
    "投诉升级": [
        "客服一直不处理我要正式投诉", "再不解决我就找平台介入", "你们的处理态度太差了",
        "我要投诉商家虚假宣传", "问题拖了很久要求主管回复", "请给我明确的投诉渠道",
        "这件事必须升级处理", "我要向有关部门反映",
    ],
    "其他问题": [
        "你们人工客服几点在线", "发票在哪里申请", "怎么修改收货备注",
        "能开增值税发票吗", "如何联系在线客服", "订单记录在哪里下载",
        "隐私政策在哪里查看", "怎么给商品评价",
    ],
}

EVAL_TEMPLATES = {
    "商品咨询": ["想确认{product}是否兼容我的设备", "下单前能介绍一下{product}的参数吗", "这件商品有别的颜色可选吗"],
    "退款退货": ["东西不符合预期，我准备退掉", "能撤销这笔购买并把钱退回来吗", "换一个尺码需要走什么流程"],
    "物流异常": ["运输轨迹很久没有变化", "系统说已送达，家里却没有包裹", "我的件是不是在途中丢失了"],
    "商品质量": ["拆箱发现商品有裂缝", "用了不到一天就无法启动", "收到的货少了关键零件"],
    "支付问题": ["钱已经划走，系统却让我重新付款", "结算阶段总是无法完成交易", "同一笔订单被收了两次费用"],
    "优惠活动": ["结算时没有享受到页面宣传的折扣", "领取的券在付款时无法选择", "活动承诺的赠品没有出现在订单里"],
    "账号问题": ["登录时总收不到短信", "账户被陌生设备访问了", "安全验证导致我无法继续购买"],
    "投诉升级": ["普通客服无法解决，请转交负责人", "我要求登记正式客诉并给出编号", "长期不处理我会寻求外部渠道帮助"],
    "其他问题": ["电子发票从哪里下载", "人工服务的工作时间是什么", "我想修改订单上的备注信息"],
}

COMPATIBLE_PAIRS = [
    ("物流异常", "退款退货"), ("商品质量", "退款退货"), ("商品质量", "投诉升级"),
    ("物流异常", "投诉升级"), ("支付问题", "投诉升级"), ("优惠活动", "支付问题"),
    ("商品咨询", "优惠活动"), ("账号问题", "支付问题"), ("退款退货", "投诉升级"),
    ("商品咨询", "其他问题"),
]


def _render(template: str, rng: random.Random) -> str:
    return template.format(product=rng.choice(PRODUCTS), color=rng.choice(COLORS))


def _decorate(text: str, rng: random.Random) -> str:
    prefix = rng.choice(PREFIXES)
    suffix = rng.choice(SUFFIXES)
    order_hint = rng.choice(["", "，订单尾号**%04d" % rng.randint(0, 9999), "，刚刚下的单"])
    punctuation = rng.choice(["。", "！", "？", ""])
    return re.sub(r"\s+", "", f"{prefix}{text}{order_hint}{punctuation}{suffix}")


def _one_record(split: str, rng: random.Random, multi_label_rate: float) -> tuple[str, list[str]]:
    templates = TRAIN_TEMPLATES if split == "train" else EVAL_TEMPLATES
    if rng.random() < multi_label_rate:
        labels = list(rng.choice(COMPATIBLE_PAIRS))
        first = _render(rng.choice(templates[labels[0]]), rng)
        second = _render(rng.choice(templates[labels[1]]), rng)
        text = first + rng.choice(CONNECTORS) + second
    else:
        labels = [rng.choice(LABELS)]
        text = _render(rng.choice(templates[labels[0]]), rng)
    return _decorate(text, rng), labels


def generate_split(
    split: str,
    size: int,
    seed: int,
    multi_label_rate: float,
    excluded_texts: set[str] | None = None,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    seen: set[str] = set(excluded_texts or set())
    records: list[dict[str, object]] = []
    attempts = 0
    while len(records) < size and attempts < size * 30:
        attempts += 1
        text, labels = _one_record(split, rng, multi_label_rate)
        if text in seen:
            continue
        seen.add(text)
        records.append(
            {
                "id": f"{split}-{len(records)+1:05d}",
                "text": text,
                "labels": labels,
                "split": split,
                "is_synthetic": True,
            }
        )
    if len(records) < size:
        raise RuntimeError(f"Could only generate {len(records)} unique records for {split}")
    return records


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_dataset(output_dir: Path = DATA_DIR, seed: int = 20260721) -> dict[str, object]:
    config = {
        "train": (1800, 0.35),
        "validation": (300, 0.40),
        "test": (300, 0.40),
    }
    summary: dict[str, object] = {"seed": seed, "splits": {}}
    all_texts: set[str] = set()
    for offset, (split, (size, rate)) in enumerate(config.items()):
        records = generate_split(split, size, seed + offset, rate, excluded_texts=all_texts)
        overlap = all_texts.intersection(str(record["text"]) for record in records)
        if overlap:
            raise RuntimeError(f"Unexpected cross-split duplicates: {len(overlap)}")
        all_texts.update(str(record["text"]) for record in records)
        write_jsonl(output_dir / f"{split}.jsonl", records)
        counts = Counter(label for record in records for label in record["labels"])
        summary["splits"][split] = {
            "records": len(records),
            "multi_label_records": sum(len(record["labels"]) > 1 for record in records),
            "label_counts": dict(counts),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic intent dataset")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    summary = build_dataset(args.output_dir, args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
