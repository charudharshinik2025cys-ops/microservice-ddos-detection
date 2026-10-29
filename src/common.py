"""Shared paths, schema, and configuration helpers."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FEATURES = ["request_rate", "avg_latency_ms", "error_rate", "timeout_rate", "cpu_percent", "memory_percent"]
CLASSES = ["normal", "directly_targeted", "indirectly_affected"]


def load_config():
    with (ROOT / "config.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def ensure_dirs():
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
