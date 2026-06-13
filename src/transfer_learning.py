"""
Cross-Subject Transfer Learning（A04 / A06 专项）

实验:
  1. Cross-Subject CSP+SVM
  2. Cross-Subject Riemannian (Cov → TangentSpace → LR)
  3. Fine-Tuning（Source 预训练 + Target 20% 微调，80% 测试）
  4. EEGNet Transfer（需 torch）
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from mne.decoding import CSP
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import GDF_CSP_COMPONENTS, GDF_DIR, RANDOM_STATE, TEST_SIZE
from src.gdf_loader import subject_to_gdf_path
from src.gdf_preprocessing import load_mi_epochs_flexible

# 统一跨受试者预处理（不做单人调参）
UNIFIED_PREPROCESS = {
    "bandpass": (8.0, 30.0),
    "tmin": 0.5,
    "tmax": 2.5,
    "car": False,
}

TRANSFER_CONFIGS = {
    "A04": {
        "target": "A04",
        "sources": ["A01", "A02", "A03", "A05", "A07", "A08", "A09"],
        "baseline_cv": 0.503,
    },
    "A06": {
        "target": "A06",
        "sources": ["A01", "A02", "A03", "A04", "A05", "A07", "A08", "A09"],
        "baseline_cv": 0.490,
    },
}


def load_subject_epochs(subject: str, gdf_dir=None) -> Tuple[np.ndarray, np.ndarray]:
    """加载单个受试者 epoch（统一配置）。"""
    cfg = UNIFIED_PREPROCESS
    path = subject_to_gdf_path(subject, session="T", gdf_dir=gdf_dir or GDF_DIR)
    X, y, _, _, _ = load_mi_epochs_flexible(
        path,
        tmin=cfg["tmin"],
        tmax=cfg["tmax"],
        bandpass=cfg["bandpass"],
        apply_car_ref=cfg["car"],
    )
    return X, y


def load_pooled_sources(sources: List[str], gdf_dir=None) -> Tuple[np.ndarray, np.ndarray]:
    """合并多个 Source 受试者全部 trial。"""
    X_list, y_list = [], []
    for subj in sources:
        X, y = load_subject_epochs(subj, gdf_dir)
        X_list.append(X)
        y_list.append(y)
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def split_target_finetune(
    X_tgt: np.ndarray,
    y_tgt: np.ndarray,
    finetune_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Target 分层划分：20% 微调 / 80% 测试。"""
    return train_test_split(
        X_tgt, y_tgt,
        test_size=1.0 - finetune_ratio,
        random_state=RANDOM_STATE,
        stratify=y_tgt,
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def build_csp_svm() -> Pipeline:
    return Pipeline([
        ("csp", CSP(n_components=GDF_CSP_COMPONENTS, reg="ledoit_wolf", log=True)),
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", random_state=RANDOM_STATE)),
    ])


def build_riemannian() -> Pipeline:
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace

    return Pipeline([
        ("cov", Covariances(estimator="lwf")),
        ("ts", TangentSpace()),
        ("lr", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])


def exp1_cross_subject_csp_svm(
    X_src: np.ndarray, y_src: np.ndarray,
    X_tgt: np.ndarray, y_tgt: np.ndarray,
) -> Dict[str, float]:
    """Source 训练 → Target 全量测试。"""
    pipe = build_csp_svm()
    pipe.fit(X_src, y_src)
    pred = pipe.predict(X_tgt)
    return _metrics(y_tgt, pred)


def exp2_cross_subject_riemannian(
    X_src: np.ndarray, y_src: np.ndarray,
    X_tgt: np.ndarray, y_tgt: np.ndarray,
) -> Dict[str, float]:
    """Source 训练 → Target 全量测试。"""
    pipe = build_riemannian()
    pipe.fit(X_src, y_src)
    pred = pipe.predict(X_tgt)
    return _metrics(y_tgt, pred)


def exp3_finetune_csp_svm(
    X_src: np.ndarray, y_src: np.ndarray,
    X_ft: np.ndarray, y_ft: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    method: str = "csp_svm",
) -> Dict[str, float]:
    """
    Source 预训练 → 合并 Source+Target20% 微调 → Target80% 测试。
    """
    if method == "csp_svm":
        pipe = build_csp_svm()
    else:
        pipe = build_riemannian()

    pipe.fit(X_src, y_src)
    X_combined = np.concatenate([X_src, X_ft], axis=0)
    y_combined = np.concatenate([y_src, y_ft], axis=0)
    pipe.fit(X_combined, y_combined)
    pred = pipe.predict(X_test)
    return _metrics(y_test, pred)


def exp4_eegnet_transfer(
    X_src: np.ndarray, y_src: np.ndarray,
    X_ft: np.ndarray, y_ft: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> Dict[str, float]:
    """EEGNet: Source 预训练 → Target 20% 微调 → 80% 测试。"""
    from src.eegnet import EEGNetClassifier

    clf = EEGNetClassifier(n_epochs_train=80, random_state=RANDOM_STATE)
    clf.fit(X_src, y_src)
    X_combined = np.concatenate([X_src, X_ft], axis=0)
    y_combined = np.concatenate([y_src, y_ft], axis=0)
    clf.finetune(X_combined, y_combined, n_epochs_finetune=40, learning_rate_finetune=5e-4)
    pred = clf.predict(X_test)
    return _metrics(y_test, pred)


def run_transfer_for_target(
    target_key: str,
    gdf_dir=None,
) -> List[Dict]:
    """对单个 Target 运行全部迁移实验。"""
    cfg = TRANSFER_CONFIGS[target_key]
    target = cfg["target"]
    sources = cfg["sources"]
    baseline = cfg["baseline_cv"]

    print(f"\n{'=' * 55}")
    print(f"  Target: {target}  |  Sources: {', '.join(sources)}")
    print(f"  基线 (within-subject CV): {baseline:.1%}")
    print(f"{'=' * 55}")

    X_src, y_src = load_pooled_sources(sources, gdf_dir)
    X_tgt, y_tgt = load_subject_epochs(target, gdf_dir)
    X_ft, X_test, y_ft, y_test = split_target_finetune(X_tgt, y_tgt)

    print(f"  Source trials: {len(y_src)}  |  Target: {len(y_tgt)} (finetune {len(y_ft)}, test {len(y_test)})")

    rows = []

    # Exp 1
    m1 = exp1_cross_subject_csp_svm(X_src, y_src, X_tgt, y_tgt)
    rows.append(_result_row(
        target, sources, "exp1_cross_subject_csp_svm", m1, baseline,
        "source_train→target_test", len(y_src), 0, len(y_tgt),
    ))
    print(f"  [Exp1] CSP+SVM transfer: {m1['accuracy']:.1%}  Δ={((m1['accuracy']-baseline)*100):+.1f}pp")

    # Exp 2
    m2 = exp2_cross_subject_riemannian(X_src, y_src, X_tgt, y_tgt)
    rows.append(_result_row(
        target, sources, "exp2_cross_subject_riemannian", m2, baseline,
        "Cov→TS→LR", len(y_src), 0, len(y_tgt),
    ))
    print(f"  [Exp2] Riemannian transfer: {m2['accuracy']:.1%}  Δ={((m2['accuracy']-baseline)*100):+.1f}pp")

    # Exp 3a - CSP+SVM fine-tune
    m3a = exp3_finetune_csp_svm(X_src, y_src, X_ft, y_ft, X_test, y_test, method="csp_svm")
    rows.append(_result_row(
        target, sources, "exp3_finetune_csp_svm", m3a, baseline,
        "source+target20%→target80%", len(y_src), len(y_ft), len(y_test),
    ))
    print(f"  [Exp3] CSP+SVM fine-tune: {m3a['accuracy']:.1%}  Δ={((m3a['accuracy']-baseline)*100):+.1f}pp")

    # Exp 3b - Riemannian fine-tune
    m3b = exp3_finetune_csp_svm(X_src, y_src, X_ft, y_ft, X_test, y_test, method="riemannian")
    rows.append(_result_row(
        target, sources, "exp3_finetune_riemannian", m3b, baseline,
        "source+target20%→target80%", len(y_src), len(y_ft), len(y_test),
    ))
    print(f"  [Exp3] Riemannian fine-tune: {m3b['accuracy']:.1%}  Δ={((m3b['accuracy']-baseline)*100):+.1f}pp")

    # Exp 4 - EEGNet
    try:
        import torch  # noqa: F401
        m4 = exp4_eegnet_transfer(X_src, y_src, X_ft, y_ft, X_test, y_test)
        rows.append(_result_row(
            target, sources, "exp4_eegnet_transfer", m4, baseline,
            "pretrain+finetune→target80%", len(y_src), len(y_ft), len(y_test),
        ))
        print(f"  [Exp4] EEGNet transfer: {m4['accuracy']:.1%}  Δ={((m4['accuracy']-baseline)*100):+.1f}pp")
    except ImportError:
        print("  [Exp4] EEGNet: 跳过（torch 未安装）")
        rows.append({
            "target": target,
            "sources": ",".join(sources),
            "experiment": "exp4_eegnet_transfer",
            "config": "skipped_no_torch",
            "accuracy": np.nan,
            "kappa": np.nan,
            "baseline_cv": baseline,
            "improvement_pp": np.nan,
            "n_source_trials": len(y_src),
            "n_target_finetune": len(y_ft),
            "n_target_test": len(y_test),
        })

    return rows


def _result_row(
    target: str,
    sources: List[str],
    experiment: str,
    metrics: Dict[str, float],
    baseline: float,
    config: str,
    n_src: int = 0,
    n_ft: int = 0,
    n_test: int = 0,
) -> Dict:
    acc = metrics["accuracy"]
    return {
        "target": target,
        "sources": ",".join(sources),
        "experiment": experiment,
        "config": config,
        "accuracy": acc,
        "kappa": metrics["kappa"],
        "baseline_cv": baseline,
        "improvement_pp": (acc - baseline) * 100,
        "n_source_trials": n_src,
        "n_target_finetune": n_ft,
        "n_target_test": n_test,
    }
