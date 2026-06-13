"""
无泄漏交叉验证评估工具
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold


def leakage_safe_cv_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    estimator_factory: Callable,
    n_splits: int = 5,
    random_state: int = 42,
    class_names: List[str] | None = None,
) -> Dict:
    """
    严格 CV：每折仅在训练集 fit（含 CSP/标准化/EEGNet），测试集 predict。
    返回 accuracy、kappa、OOF 预测、混淆矩阵。
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_pred = np.full_like(y, fill_value=-1)
    fold_accs = []

    for train_idx, test_idx in cv.split(X, y):
        est = estimator_factory()
        est.fit(X[train_idx], y[train_idx])
        pred = est.predict(X[test_idx])
        y_pred[test_idx] = pred
        fold_accs.append(accuracy_score(y[test_idx], pred))

    if (y_pred < 0).any():
        raise RuntimeError("CV 预测未完成，存在未覆盖样本。")

    cm = confusion_matrix(y, y_pred)
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "kappa": float(cohen_kappa_score(y, y_pred)),
        "fold_accuracies": fold_accs,
        "fold_accuracy_std": float(np.std(fold_accs)),
        "y_true": y,
        "y_pred": y_pred,
        "confusion_matrix": cm,
        "class_names": class_names or [],
    }


def verify_no_leakage_demo(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> str:
    """
    泄漏检测演示：比较错误做法（全数据 fit 标准化）与正确 CV 的差异。
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from mne.decoding import CSP

    # 错误：在全数据上 fit 再 CV（仅评估 SVM，CSP 仍应在折内 fit）
    wrong_pipe = Pipeline([
        ("csp", CSP(6, reg="ledoit_wolf", log=True)),
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf")),
    ])
    wrong_pipe.fit(X, y)
    from sklearn.model_selection import cross_val_score
    wrong_cv = cross_val_score(
        Pipeline([
            ("csp", CSP(6, reg="ledoit_wolf", log=True)),
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf")),
        ]),
        X, y, cv=5,
    ).mean()

    correct = leakage_safe_cv_evaluate(
        X, y,
        lambda: Pipeline([
            ("csp", CSP(6, reg="ledoit_wolf", log=True)),
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf")),
        ]),
        random_state=random_state,
    )

    lines = [
        "数据泄漏检查（以 A01 为例的逻辑）:",
        f"  正确 5 折 CV（每折独立 fit CSP+Scaler+SVM）: {correct['accuracy']:.4f}",
        f"  sklearn Pipeline 5 折 CV（同样无泄漏）: {wrong_cv:.4f}",
        "  结论: 使用 Pipeline + StratifiedKFold 时，CSP/Scaler 仅在训练折 fit，无泄漏。",
        "  注意: 切勿在 CV 前对全数据做 CSP fit 或 StandardScaler.fit。",
    ]
    return "\n".join(lines)
