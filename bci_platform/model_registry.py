"""模型热加载注册表（单例，只 load 一次）。"""
from __future__ import annotations

import pickle
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from config import BEST_MODEL_PATH, MODELS_DIR


class ModelRegistry:
    """线程安全的模型缓存。"""

    _instance: Optional["ModelRegistry"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def get(cls) -> "ModelRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def load(self, model_path: Path | str | None = None) -> Dict[str, Any]:
        path = Path(model_path or BEST_MODEL_PATH).resolve()
        key = str(path)
        if key not in self._cache:
            if not path.exists():
                raise FileNotFoundError(f"模型不存在: {path}")
            with open(path, "rb") as f:
                self._cache[key] = pickle.load(f)
        return self._cache[key]

    def clear(self) -> None:
        self._cache.clear()

    @staticmethod
    def list_available_models() -> Dict[str, Path]:
        """列出 models/ 下所有 motor_imagery_*.pkl。"""
        models = {}
        universal = MODELS_DIR / "motor_imagery_universal.pkl"
        if universal.exists():
            models["UNIVERSAL"] = universal
        for p in sorted(MODELS_DIR.glob("motor_imagery_a*.pkl")):
            name = p.stem.replace("motor_imagery_", "").upper()
            models[name] = p
        default = MODELS_DIR / "motor_imagery_model.pkl"
        if default.exists():
            models["DEFAULT"] = default
        return models
