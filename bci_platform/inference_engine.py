"""推理引擎：单条 / 批量预测，复用已保存 pipeline。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from bci_platform.feature_pipeline import GDFReplaySource, config_from_bundle
from bci_platform.logging_config import setup_inference_logger
from bci_platform.model_registry import ModelRegistry
from config import LABEL_DISPLAY_NAMES
from src.universal_model import resolve_sub_bundle


@dataclass
class PredictionResult:
    label: str
    display_name: str
    command: str
    confidence: Optional[float] = None
    true_label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "label": self.label,
            "display_name": self.display_name,
            "command": self.command,
        }
        if self.confidence is not None:
            d["confidence"] = self.confidence
        if self.true_label is not None:
            d["true_label"] = self.true_label
        return d


class InferenceEngine:
    """
    本地推理服务。
    - 模型热加载（ModelRegistry 单例）
    - 支持单 trial / batch / GDF replay
    """

    COMMAND_MAP = {
        "left": "Move Left",
        "right": "Move Right",
        "foot": "Move Forward",
        "tongue": "Select",
    }

    def __init__(self, model_path: Path | str | None = None):
        self.registry = ModelRegistry.get()
        self.bundle = self.registry.load(model_path)
        self.pipeline = self.bundle["pipeline"]
        self.label_encoder = self.bundle["label_encoder"]
        self.class_names = self.bundle["class_names"]
        self.is_universal = self.bundle.get("type") == "universal_router"
        self.config = config_from_bundle(self.bundle)
        self._active_bundle = self.bundle
        self._routed_subject = self.bundle.get("subject", "?")
        self.logger = setup_inference_logger()
        self.logger.info(
            "Model loaded: subject=%s method=%s path=%s universal=%s",
            self.bundle.get("subject"),
            self.bundle.get("model_name"),
            model_path or "default",
            self.is_universal,
        )

    def set_gdf_context(self, gdf_path: Path | str | None = None, subject: str | None = None) -> str:
        """通用模型：根据 GDF 路径切换子模型。"""
        if not self.is_universal:
            return self.bundle.get("subject", "?")
        active, routed = resolve_sub_bundle(self.bundle, gdf_path, subject)
        self._active_bundle = active
        self._routed_subject = routed
        if hasattr(self.pipeline, "set_context"):
            self.pipeline.set_context(gdf_path=gdf_path, subject=subject)
        self.config = config_from_bundle(active)
        self.logger.info("Universal route -> %s (%s)", routed, active.get("model_name"))
        return routed

    def _decode(self, pred_idx: int) -> PredictionResult:
        raw = self.label_encoder.inverse_transform([pred_idx])[0]
        display = LABEL_DISPLAY_NAMES.get(raw, raw)
        command = self.COMMAND_MAP.get(raw, "Unknown")
        return PredictionResult(label=raw, display_name=display, command=command)

    def predict_one(self, epoch: np.ndarray) -> PredictionResult:
        """单 trial 推理。epoch shape: (n_channels, n_times)。"""
        x = np.asarray(epoch)
        if x.ndim == 2:
            x = x[np.newaxis, ...]
        pred = int(self.pipeline.predict(x)[0])
        result = self._decode(pred)
        self.logger.info("predict_one | label=%s command=%s", result.label, result.command)
        return result

    def predict_batch(
        self,
        epochs: np.ndarray,
        true_labels: Optional[np.ndarray] = None,
    ) -> List[PredictionResult]:
        """批量推理。epochs shape: (n_trials, n_channels, n_times)。"""
        epochs = np.asarray(epochs)
        preds = self.pipeline.predict(epochs)
        results = []
        for i, p in enumerate(preds):
            r = self._decode(int(p))
            if true_labels is not None and i < len(true_labels):
                true_raw = self.label_encoder.inverse_transform([int(true_labels[i])])[0]
                r.true_label = true_raw
            results.append(r)
            self.logger.info(
                "predict_batch[%d] | pred=%s true=%s",
                i, r.label, r.true_label or "-",
            )
        return results

    def create_replay_source(
        self,
        subject_override: str | None = None,
        gdf_path: Path | str | None = None,
        shuffle: bool = False,
    ) -> GDFReplaySource:
        if self.is_universal:
            self.set_gdf_context(gdf_path, subject_override)
        else:
            subject_override = subject_override or self.config.get("subject")
        return GDFReplaySource(
            self.bundle,
            subject_override=subject_override,
            gdf_path=gdf_path,
            shuffle=shuffle,
        )

    def replay_all(self, batch_size: int = 32) -> Dict[str, Any]:
        """GDF 全量 replay，返回准确率统计。"""
        source = self.create_replay_source()
        all_pred, all_true = [], []
        for bx, by in source.iter_all(batch_size):
            preds = self.pipeline.predict(bx)
            all_pred.extend(preds.tolist())
            all_true.extend(by.tolist())
        acc = float(np.mean(np.array(all_pred) == np.array(all_true)))
        self.logger.info("replay_all | trials=%d accuracy=%.2f%%", len(all_true), acc * 100)
        return {
            "n_trials": len(all_true),
            "accuracy": acc,
            "subject": self.config["subject"],
            "predictions": all_pred,
            "true_labels": all_true,
        }
