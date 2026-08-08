from collections import Counter

from app.labels import LABELS
from ml.data_generator import generate_split


def test_generator_is_reproducible_and_covers_all_labels() -> None:
    first = generate_split("train", 180, seed=42, multi_label_rate=0.35)
    second = generate_split("train", 180, seed=42, multi_label_rate=0.35)
    assert first == second
    counts = Counter(label for record in first for label in record["labels"])
    assert set(counts) == set(LABELS)
    assert len({record["text"] for record in first}) == len(first)
    assert any(len(record["labels"]) > 1 for record in first)


def test_evaluation_wording_differs_from_training_templates() -> None:
    train = generate_split("train", 90, seed=1, multi_label_rate=0.0)
    test = generate_split("test", 90, seed=2, multi_label_rate=0.0)
    assert not ({record["text"] for record in train} & {record["text"] for record in test})

