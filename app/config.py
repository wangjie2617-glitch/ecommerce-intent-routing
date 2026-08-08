"""Central path and runtime configuration."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
BASELINE_DIR = ARTIFACTS_DIR / "baseline"
BERT_DIR = ARTIFACTS_DIR / "bert"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_TYPE = os.getenv("MODEL_TYPE", "auto").lower()
MANUAL_REVIEW_THRESHOLD = float(os.getenv("MANUAL_REVIEW_THRESHOLD", "0.55"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "100"))

