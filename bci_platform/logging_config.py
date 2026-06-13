"""推理日志配置。"""
from __future__ import annotations

import logging
from pathlib import Path

from config import OUTPUTS_DIR

LOG_DIR = OUTPUTS_DIR / "inference_logs"


def setup_inference_logger(name: str = "bci_inference") -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(LOG_DIR / "predictions.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger
