"""
A02 / A05 / A09 高潜力受试者优化实验

任务1: 论文版 FBCSP (8 子带 + MI + LDA)
任务2: CSP + Riemannian 融合特征
任务3: XGBoost / LightGBM / ExtraTrees 分类器搜索
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import GDF_DIR, OUTPUTS_DIR, RANDOM_STATE
from src.experiment_eval import leakage_safe_cv_evaluate
from src.gdf_loader import subject_to_gdf_path
from src.gdf_preprocessing import load_mi_epochs_flexible
from src.hybrid_features import FeatureFusionExtractor, HybridCSPRiemannian
from src.paper_fbcsp import PaperFBCSP

OUT_DIR = OUTPUTS_DIR / "experiments" / "high_potential_subjects"
TARGETS = ["A02", "A05", "A09"]

BAND_MAP = {
    "4-40Hz": (4.0, 40.0),
    "7-35Hz": (7.0, 35.0),
    "8-30Hz": (8.0, 30.0),
}
WINDOW_MAP = {
    "0.5-2.5s": (0.5, 2.5),
    "0.5-3.5s": (0.5, 3.5),
    "0.5-4.0s": (0.5, 4.0),
    "1.0-4.0s": (1.0, 4.0),
}

# 各受试者当前最优 CV（432 网格）
CURRENT_BEST = {
    "A02": 0.6875,
    "A05": 0.7500,
    "A09": 0.7604,
}
OVERALL_CURRENT = 0.7272  # 9 人平均

# 全 9 人当前 CV（用于重算整体平均）
ALL_SUBJECTS_CV = {
    "A01": 0.8229, "A02": 0.6875, "A03": 0.8681, "A04": 0.5035,
    "A05": 0.7500, "A06": 0.4896, "A07": 0.7951, "A08": 0.8681, "A09": 0.7604,
}

# 每人搜索配置（基于先前最优 + 邻近变体）
SUBJECT_CONFIGS: Dict[str, List[Dict]] = {
    "A02": [
        {"band": "7-35Hz", "window": "1.0-4.0s", "car": False},
        {"band": "7-35Hz", "window": "0.5-4.0s", "car": False},
        {"band": "4-40Hz", "window": "1.0-4.0s", "car": False},
        {"band": "7-35Hz", "window": "1.0-4.0s", "car": True},
    ],
    "A05": [
        {"band": "4-40Hz", "window": "0.5-4.0s", "car": True},
        {"band": "4-40Hz", "window": "0.5-4.0s", "car": False},
        {"band": "7-35Hz", "window": "0.5-4.0s", "car": True},
        {"band": "4-40Hz", "window": "0.5-3.5s", "car": True},
    ],
    "A09": [
        {"band": "7-35Hz", "window": "0.5-2.5s", "car": True},
        {"band": "7-35Hz", "window": "0.5-3.5s", "car": True},
        {"band": "4-40Hz", "window": "0.5-2.5s", "car": True},
        {"band": "7-35Hz", "window": "0.5-2.5s", "car": False},
    ],
}


def load_epochs(subject: str, cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:
    path = subject_to_gdf_path(subject, session="T", gdf_dir=GDF_DIR)
    bandpass = BAND_MAP[cfg["band"]]
    tmin, tmax = WINDOW_MAP[cfg["window"]]
    # 论文 FBCSP 需宽频原始 epoch（子带内部滤波）
    X, y, _, _, _ = load_mi_epochs_flexible(
        path, tmin=tmin, tmax=tmax, bandpass=None, apply_car_ref=cfg["car"]
    )
    return X, y


def _cv_row(subject: str, task: str, cfg: Dict, metrics: Dict, current_cv: float) -> Dict:
    acc = metrics["accuracy"]
    return {
        "subject": subject,
        "task": task,
        "band": cfg["band"],
        "window": cfg["window"],
        "car": cfg["car"],
        "accuracy": acc,
        "kappa": metrics["kappa"],
        "fold_accuracy_std": metrics["fold_accuracy_std"],
        "current_cv": current_cv,
        "improvement_pp": (acc - current_cv) * 100,
    }


def run_task1_fbcsp(subject: str) -> Tuple[pd.DataFrame, Dict]:
    """论文版 FBCSP。"""
    current = CURRENT_BEST[subject]
    rows = []
    best = {"accuracy": 0.0}

    for cfg in SUBJECT_CONFIGS[subject]:
        X, y = load_epochs(subject, cfg)
        metrics = leakage_safe_cv_evaluate(
            X, y,
            lambda: PaperFBCSP(top_k_features=22, sample_rate=250.0),
            random_state=RANDOM_STATE,
        )
        rows.append(_cv_row(subject, "paper_fbcsp", cfg, metrics, current))
        if metrics["accuracy"] > best.get("accuracy", 0):
            best = {**cfg, **metrics, "task": "paper_fbcsp"}

    return pd.DataFrame(rows), best


def run_task2_hybrid(subject: str) -> Tuple[pd.DataFrame, Dict]:
    """CSP + Riemannian 融合。"""
    current = CURRENT_BEST[subject]
    rows = []
    best = {"accuracy": 0.0}

    for cfg in SUBJECT_CONFIGS[subject]:
        X, y = load_epochs(subject, cfg)
        metrics = leakage_safe_cv_evaluate(
            X, y,
            lambda: HybridCSPRiemannian(top_k_fbcsp=22, sample_rate=250.0),
            random_state=RANDOM_STATE,
        )
        rows.append(_cv_row(subject, "hybrid_csp_riemannian", cfg, metrics, current))
        if metrics["accuracy"] > best.get("accuracy", 0):
            best = {**cfg, **metrics, "task": "hybrid_csp_riemannian"}

    return pd.DataFrame(rows), best


def _build_classifiers() -> Dict:
    clfs = {
        "LDA": lambda: LinearDiscriminantAnalysis(),
        "SVM-RBF": lambda: SVC(kernel="rbf", random_state=RANDOM_STATE),
        "XGBoost": lambda: XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=RANDOM_STATE, eval_metric="mlogloss",
            verbosity=0,
        ),
        "ExtraTrees": lambda: ExtraTreesClassifier(
            n_estimators=300, max_depth=None, random_state=RANDOM_STATE,
        ),
    }
    try:
        from lightgbm import LGBMClassifier
        clfs["LightGBM"] = lambda: LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=RANDOM_STATE, verbose=-1,
        )
    except ImportError:
        pass
    return clfs


class FusionClassifierPipeline:
    """融合特征 + 分类器（sklearn 风格）。"""

    def __init__(self, classifier_factory):
        self.classifier_factory = classifier_factory

    def fit(self, X, y):
        self.extractor_ = FeatureFusionExtractor(top_k_fbcsp=22, sample_rate=250.0)
        self.extractor_.fit(X, y)
        feats = self.extractor_.transform(X)
        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(feats)
        self.clf_ = self.classifier_factory()
        self.clf_.fit(Xs, y)
        return self

    def predict(self, X):
        feats = self.extractor_.transform(X)
        return self.clf_.predict(self.scaler_.transform(feats))


def run_task3_classifier_search(subject: str, best_cfg: Dict) -> Tuple[pd.DataFrame, Dict]:
    """在最佳预处理配置上搜索分类器。"""
    current = CURRENT_BEST[subject]
    cfg = {k: best_cfg[k] for k in ("band", "window", "car") if k in best_cfg}
    X, y = load_epochs(subject, cfg)
    rows = []
    best = {"accuracy": 0.0}
    clfs = _build_classifiers()

    for name, factory in clfs.items():
        metrics = leakage_safe_cv_evaluate(
            X, y,
            lambda f=factory: FusionClassifierPipeline(f),
            random_state=RANDOM_STATE,
        )
        row = _cv_row(subject, f"classifier_{name}", cfg, metrics, current)
        row["classifier"] = name
        rows.append(row)
        if metrics["accuracy"] > best.get("accuracy", 0):
            best = {**cfg, **metrics, "task": f"classifier_{name}", "classifier": name}

    return pd.DataFrame(rows), best


def main() -> None:
    try:
        from lightgbm import LGBMClassifier  # noqa: F401
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm", "-q"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_fbcsp, all_hybrid, all_clf = [], [], []
    subject_best: Dict[str, Dict] = {}

    print("=" * 55)
    print("  A02 / A05 / A09 高潜力受试者优化")
    print("=" * 55)

    for subject in TARGETS:
        current = CURRENT_BEST[subject]
        print(f"\n--- {subject} (当前 CV {current:.1%}) ---")

        df1, b1 = run_task1_fbcsp(subject)
        all_fbcsp.append(df1)
        print(f"  [Task1] Paper FBCSP 最佳: {b1['accuracy']:.1%} ({b1['band']} {b1['window']} CAR={b1['car']})")

        df2, b2 = run_task2_hybrid(subject)
        all_hybrid.append(df2)
        print(f"  [Task2] Hybrid 最佳: {b2['accuracy']:.1%}")

        # Task3 使用 Task1/2 中更优的预处理配置
        best_cfg = b1 if b1["accuracy"] >= b2["accuracy"] else b2
        df3, b3 = run_task3_classifier_search(subject, best_cfg)
        all_clf.append(df3)
        print(f"  [Task3] Classifier 最佳: {b3['classifier']} → {b3['accuracy']:.1%}")

        # 三人各自全局最佳
        overall_best = max([b1, b2, b3], key=lambda x: x["accuracy"])
        subject_best[subject] = overall_best
        print(f"  >>> {subject} 总最佳: {overall_best['accuracy']:.1%} "
              f"(Δ={((overall_best['accuracy']-current)*100):+.1f}pp)")

    fbcsp_df = pd.concat(all_fbcsp, ignore_index=True)
    hybrid_df = pd.concat(all_hybrid, ignore_index=True)
    clf_df = pd.concat(all_clf, ignore_index=True)

    fbcsp_df.to_csv(OUT_DIR / "real_fbcsp_results.csv", index=False)
    hybrid_df.to_csv(OUT_DIR / "hybrid_feature_results.csv", index=False)
    clf_df.to_csv(OUT_DIR / "classifier_search_results.csv", index=False)

    # 汇总表
    summary_rows = []
    for subject in TARGETS:
        cur = CURRENT_BEST[subject]
        best = subject_best[subject]
        summary_rows.append({
            "subject": subject,
            "current_cv": cur,
            "best_cv": best["accuracy"],
            "improvement_pp": (best["accuracy"] - cur) * 100,
            "best_task": best.get("task", ""),
            "best_config": f"{best.get('band','')} {best.get('window','')} CAR={best.get('car','')}",
            "best_classifier": best.get("classifier", "LDA/hybrid"),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "subject_summary.csv", index=False)

    # 新整体平均 CV
    new_all = dict(ALL_SUBJECTS_CV)
    for subject in TARGETS:
        new_all[subject] = subject_best[subject]["accuracy"]
    new_mean = np.mean(list(new_all.values()))
    old_mean = np.mean(list(ALL_SUBJECTS_CV.values()))

    overall = pd.DataFrame([{
        "old_mean_cv_9subjects": old_mean,
        "new_mean_cv_9subjects": new_mean,
        "improvement_pp": (new_mean - old_mean) * 100,
        "target_78_reached": new_mean >= 0.78,
    }])
    overall.to_csv(OUT_DIR / "overall_mean_cv.csv", index=False)

    per_subject = pd.DataFrame([
        {"subject": s, "old_cv": ALL_SUBJECTS_CV[s], "new_cv": new_all[s]}
        for s in sorted(new_all)
    ])
    per_subject.to_csv(OUT_DIR / "all_subjects_updated_cv.csv", index=False)

    print(f"\n{'=' * 55}")
    print("  汇总")
    print(f"{'=' * 55}")
    print(summary_df.to_string(index=False))
    print(f"\n  9 人平均 CV: {old_mean:.1%} → {new_mean:.1%} "
          f"(Δ={((new_mean-old_mean)*100):+.1f}pp, 目标78%: {'达成' if new_mean>=0.78 else '未达成'})")
    print(f"\n  输出: {OUT_DIR}")


if __name__ == "__main__":
    main()
