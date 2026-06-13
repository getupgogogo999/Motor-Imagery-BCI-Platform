"""特征 / 数据加载流水线（与模型解耦）。"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np

from config import GDF_DIR
from src.gdf_loader import subject_to_gdf_path
from src.gdf_preprocessing import load_epochs_auto
from src.universal_model import resolve_sub_bundle

BAND_MAP = {
    "8-30Hz": (8.0, 30.0),
    "7-35Hz": (7.0, 35.0),
    "4-40Hz": (4.0, 40.0),
}
WINDOW_MAP = {
    "0.5-2.5s": (0.5, 2.5),
    "0.5-3.5s": (0.5, 3.5),
    "0.5-4.0s": (0.5, 4.0),
    "1.0-4.0s": (1.0, 4.0),
}


def config_from_bundle(bundle: Dict) -> Dict:
    """从已保存模型包读取预处理配置。"""
    cfg = bundle.get("config") or {}
    return {
        "band": cfg.get("band", "8-30Hz"),
        "window": cfg.get("window", "0.5-2.5s"),
        "car": bool(cfg.get("car", False)),
        "subject": bundle.get("subject", "A01"),
        "gdf_format": cfg.get("gdf_format", "auto"),
    }


def load_epochs_for_bundle(
    bundle: Dict,
    gdf_dir: Path | None = None,
    subject_override: str | None = None,
    gdf_path: Path | str | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """按模型包内配置加载 GDF epoch；可指定 gdf_path 覆盖默认路径。"""
    active_bundle, _ = resolve_sub_bundle(bundle, gdf_path, subject_override)
    cfg = config_from_bundle(active_bundle)
    subject = subject_override or cfg["subject"]
    bandpass = BAND_MAP.get(cfg["band"], (8.0, 30.0))
    tmin, tmax = WINDOW_MAP.get(cfg["window"], (0.5, 2.5))
    path = Path(gdf_path) if gdf_path else subject_to_gdf_path(subject, session="T", gdf_dir=gdf_dir or GDF_DIR)
    X, y, class_names, _, _ = load_epochs_auto(
        path,
        gdf_format=cfg.get("gdf_format", "auto"),
        tmin=tmin,
        tmax=tmax,
        bandpass=bandpass,
        apply_car_ref=cfg["car"],
    )
    return X, y, class_names


class GDFReplaySource:
    """GDF 试次迭代器，供 batch / 实时 replay 使用。"""

    def __init__(
        self,
        bundle: Dict,
        gdf_dir: Path | None = None,
        subject_override: str | None = None,
        gdf_path: Path | str | None = None,
        shuffle: bool = False,
        seed: int = 42,
    ):
        self.bundle = bundle
        self.gdf_dir = gdf_dir
        self.subject_override = subject_override
        self.gdf_path = Path(gdf_path) if gdf_path else None
        self.active_bundle, self.routed_subject = resolve_sub_bundle(
            bundle, self.gdf_path, subject_override
        )
        if bundle.get("type") == "universal_router":
            pipeline = bundle.get("pipeline")
            if pipeline is not None and hasattr(pipeline, "set_context"):
                pipeline.set_context(gdf_path=self.gdf_path, subject=subject_override)
        X, y, class_names = load_epochs_for_bundle(
            bundle, gdf_dir, subject_override, self.gdf_path
        )
        self.class_names = class_names
        indices = np.arange(len(y))
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)
        self.X = X[indices]
        self.y = y[indices]
        self._pos = 0

    def __len__(self) -> int:
        return len(self.y)

    @property
    def position(self) -> int:
        return self._pos

    def reset(self) -> None:
        self._pos = 0

    def next_batch(self, batch_size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """返回下一批 trial；不足 batch_size 时返回剩余。"""
        start = self._pos
        end = min(start + batch_size, len(self.y))
        if start >= end:
            return np.empty((0, *self.X.shape[1:])), np.empty(0, dtype=int)
        batch_x = self.X[start:end]
        batch_y = self.y[start:end]
        self._pos = end
        return batch_x, batch_y

    def iter_all(self, batch_size: int = 32) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        self.reset()
        while True:
            bx, by = self.next_batch(batch_size)
            if len(by) == 0:
                break
            yield bx, by
