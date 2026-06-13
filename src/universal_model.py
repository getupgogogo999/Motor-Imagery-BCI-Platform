"""
通用智能路由模型：单个 .pkl 内含 A01–A09 各自最优模型，按 GDF 文件名自动切换。

说明：这不是「一个权重预测所有人」的跨人模型（那种 LOSO 仅 ~39%）。
      而是「一个入口、自动配对」——在 GDF 与受试者一致时，达到各人的 CV 上限。
      A04 / A06 因 BCI 失读，上限约 50%，无法达到 70%。
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import LabelEncoder

from config import MODELS_DIR, RANDOM_STATE, TEST_SIZE
from sklearn.model_selection import train_test_split

SUBJECT_PATTERN = re.compile(r"(A010|A0[1-9])", re.IGNORECASE)
ALL_SUBJECTS = [f"A{i:02d}" for i in range(1, 10)]
EXTERNAL_SUBJECTS = ["A010"]


class UniversalPipeline:
    """sklearn 风格包装器：predict 前需 set_context(gdf_path=...)。"""

    def __init__(self, bundle: Dict[str, Any]):
        self._bundle = bundle
        self._active_subject: Optional[str] = None
        self._gdf_path: Optional[Path] = None

    def set_context(
        self,
        gdf_path: Path | str | None = None,
        subject: str | None = None,
    ) -> str:
        subj = resolve_subject(gdf_path, subject, self._bundle)
        self._active_subject = subj
        self._gdf_path = Path(gdf_path) if gdf_path else None
        return subj

    @property
    def active_subject(self) -> str:
        if self._active_subject is None:
            return self._bundle.get("default_subject", "A08")
        return self._active_subject

    def _sub_pipeline(self):
        subj = self.active_subject
        subjects = self._bundle["subjects"]
        if subj not in subjects:
            subj = self._bundle.get("default_subject", "A08")
        return subjects[subj]["pipeline"]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._sub_pipeline().predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pipe = self._sub_pipeline()
        if hasattr(pipe, "predict_proba"):
            return pipe.predict_proba(X)
        raise AttributeError("Active sub-pipeline has no predict_proba")


def detect_subject_from_path(gdf_path: Path | str | None) -> Optional[str]:
    if gdf_path is None:
        return None
    name = Path(gdf_path).stem.upper()
    m = SUBJECT_PATTERN.search(name)
    return m.group(1).upper() if m else None


def resolve_subject(
    gdf_path: Path | str | None,
    subject_hint: str | None,
    bundle: Dict[str, Any],
) -> str:
    if subject_hint:
        s = subject_hint.upper()
        if not s.startswith("A"):
            s = f"A{int(subject_hint):02d}"
        if s in bundle.get("subjects", {}):
            return s[:3]
    detected = detect_subject_from_path(gdf_path)
    if detected and detected in bundle.get("subjects", {}):
        return detected
    return bundle.get("default_subject", "A08")


def resolve_sub_bundle(
    bundle: Dict[str, Any],
    gdf_path: Path | str | None = None,
    subject_override: str | None = None,
) -> Tuple[Dict[str, Any], str]:
    """从通用包或普通包解析出实际用于预处理/推理的子 bundle。"""
    if bundle.get("type") != "universal_router":
        return bundle, bundle.get("subject", "A01")

    subj = resolve_subject(gdf_path, subject_override, bundle)
    sub = bundle["subjects"][subj]
    return sub, subj


def load_subject_bundle(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def build_universal_bundle(
    models_dir: Path | None = None,
    default_subject: str = "A08",
) -> Dict[str, Any]:
    """合并 models/motor_imagery_aXX.pkl 为单个通用包。"""
    folder = models_dir or MODELS_DIR
    subjects: Dict[str, Dict[str, Any]] = {}
    routing_metrics: Dict[str, float] = {}

    subject_list = list(ALL_SUBJECTS)
    for subj in EXTERNAL_SUBJECTS:
        if (folder / f"motor_imagery_{subj.lower()}.pkl").exists():
            subject_list.append(subj)

    for subj in subject_list:
        path = folder / f"motor_imagery_{subj.lower()}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"缺少受试者模型: {path}")
        sub = load_subject_bundle(path)
        subjects[subj] = sub
        routing_metrics[subj] = float(
            sub.get("metrics", {}).get("Accuracy")
            or sub.get("metrics", {}).get("cv_accuracy")
            or sub.get("cv_accuracy", 0.0)
        )

    ref = subjects[default_subject]
    label_encoder: LabelEncoder = ref["label_encoder"]
    class_names: List[str] = ref["class_names"]

    outer = {
        "type": "universal_router",
        "model_name": "Universal Smart Router",
        "data_source": "gdf",
        "feature_mode": "universal_router",
        "subject": "ALL",
        "default_subject": default_subject,
        "subjects": subjects,
        "routing_metrics": routing_metrics,
        "per_subject_cv": {
            subj: float(sub.get("cv_accuracy", sub.get("metrics", {}).get("Accuracy", 0)))
            for subj, sub in subjects.items()
        },
        "class_names": class_names,
        "label_encoder": label_encoder,
        "config": {
            "note": "预处理参数随路由到的子模型自动切换",
            "auto_detect": "GDF 文件名中的 A01–A09",
        },
        "limitations": {
            "A04": "~50% BCI illiteracy ceiling",
            "A06": "~50% BCI illiteracy ceiling",
            "A02": "~69–73% best known single-subject CV",
        },
    }
    outer["pipeline"] = UniversalPipeline(outer)
    return outer


def validate_universal_bundle(
    bundle: Dict[str, Any],
    gdf_dir: Path,
) -> List[Dict[str, Any]]:
    """对每个受试者：路由 + holdout 测试准确率。"""
    from src.gdf_loader import subject_to_gdf_path
    from src.gdf_preprocessing import load_mi_epochs_flexible
    from src.gdf_trainer import BAND_MAP, WINDOW_MAP

    pipeline: UniversalPipeline = bundle["pipeline"]
    rows = []

    for subj in ALL_SUBJECTS:
        gdf_path = subject_to_gdf_path(subj, session="T", gdf_dir=gdf_dir)
        sub_bundle, routed = resolve_sub_bundle(bundle, gdf_path=gdf_path)
        cfg = sub_bundle.get("config", {})
        bandpass = BAND_MAP.get(cfg.get("band", "8-30Hz"), (8.0, 30.0))
        tmin, tmax = WINDOW_MAP.get(cfg.get("window", "0.5-2.5s"), (0.5, 2.5))
        use_car = bool(cfg.get("car", False))

        X, y, _, _, _ = load_mi_epochs_flexible(
            gdf_path,
            tmin=tmin,
            tmax=tmax,
            bandpass=bandpass,
            apply_car_ref=use_car,
        )
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        pipeline.set_context(gdf_path=gdf_path)
        pred = pipeline.predict(X_test)
        acc = float((pred == y_test).mean())
        cv_ref = float(bundle["per_subject_cv"].get(subj, 0))
        rows.append(
            {
                "subject": subj,
                "routed_to": routed,
                "holdout_accuracy": acc,
                "saved_cv": cv_ref,
                "method": sub_bundle.get("model_name", "?"),
            }
        )
    return rows


def save_universal_bundle(bundle: Dict[str, Any], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(bundle, f)
