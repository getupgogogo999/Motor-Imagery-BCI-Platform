"""
A04 / A06 专项优化实验模块（泄漏安全 5 折 CV）
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
from mne.decoding import CSP
from scipy.signal import butter, filtfilt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import RANDOM_STATE
from src.experiment_eval import leakage_safe_cv_evaluate

# 432 组网格实验中的基线 CV 准确率
BASELINE_CV = {"A04": 0.503, "A06": 0.490}

# 各受试者先前最优预处理配置
SUBJECT_PREPROCESS = {
    "A04": {"bandpass": (8.0, 30.0), "tmin": 0.5, "tmax": 2.5, "car": False},
    "A06": {"bandpass": (4.0, 40.0), "tmin": 0.5, "tmax": 4.0, "car": False},
}

EXTENDED_BANDS: List[Tuple[float, float]] = [
    (4, 8),
    (8, 12),
    (12, 16),
    (16, 20),
    (20, 24),
    (24, 28),
    (28, 32),
    (32, 36),
]

CSP_DIMS = [2, 4, 6, 8]
SVM_C_VALUES = [1, 10, 100]
SVM_GAMMA_VALUES = ["scale", "auto"]


class ExtendedFBCSP(BaseEstimator, ClassifierMixin):
    """8 子带独立 CSP，特征拼接后接 LDA 或 SVM。"""

    def __init__(
        self,
        bands: List[Tuple[float, float]] | None = None,
        n_components: int = 2,
        sample_rate: float = 250.0,
        classifier: str = "lda",
        svm_c: float = 1.0,
        svm_gamma: str = "scale",
    ):
        self.bands = bands or EXTENDED_BANDS
        self.n_components = n_components
        self.sample_rate = sample_rate
        self.classifier = classifier
        self.svm_c = svm_c
        self.svm_gamma = svm_gamma

    def _filter_band(self, X: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
        nyq = self.sample_rate / 2.0
        low = max(fmin / nyq, 1e-4)
        high = min(fmax / nyq, 0.999)
        if low >= high:
            return X
        b, a = butter(4, [low, high], btype="band")
        return filtfilt(b, a, X, axis=-1)

    def _extract_features(self, X: np.ndarray, y=None, fit: bool = False) -> np.ndarray:
        if fit:
            self.csps_ = []
        feats = []
        for i, (fmin, fmax) in enumerate(self.bands):
            X_band = self._filter_band(X, fmin, fmax)
            if fit:
                csp = CSP(n_components=self.n_components, reg="ledoit_wolf", log=True)
                csp.fit(X_band, y)
                self.csps_.append(csp)
            feats.append(self.csps_[i].transform(X_band))
        return np.hstack(feats)

    def fit(self, X, y):
        features = self._extract_features(np.asarray(X), y, fit=True)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(features)
        if self.classifier == "svm":
            self.clf_ = SVC(
                kernel="rbf",
                C=self.svm_c,
                gamma=self.svm_gamma,
                random_state=RANDOM_STATE,
            )
        else:
            self.clf_ = LinearDiscriminantAnalysis()
        self.clf_.fit(X_scaled, y)
        return self

    def predict(self, X):
        features = self._extract_features(np.asarray(X), fit=False)
        return self.clf_.predict(self.scaler_.transform(features))


def _result_row(
    subject: str,
    module: str,
    config: str,
    metrics: Dict,
    baseline: float,
) -> Dict:
    acc = metrics["accuracy"]
    return {
        "subject": subject,
        "module": module,
        "config": config,
        "accuracy": acc,
        "kappa": metrics["kappa"],
        "fold_accuracy_std": metrics["fold_accuracy_std"],
        "baseline_accuracy": baseline,
        "improvement": acc - baseline,
        "improvement_pct": (acc - baseline) * 100,
    }


def run_riemannian(subject: str, X: np.ndarray, y: np.ndarray) -> Dict:
    """Covariance → Tangent Space → Logistic Regression。"""
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace

    baseline = BASELINE_CV[subject]

    def factory():
        return Pipeline([
            ("cov", Covariances(estimator="lwf")),
            ("ts", TangentSpace()),
            ("lr", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ])

    metrics = leakage_safe_cv_evaluate(X, y, factory, random_state=RANDOM_STATE)
    return _result_row(subject, "riemannian", "Cov(lwf)->TangentSpace->LR", metrics, baseline)


def run_extended_fbcsp(
    subject: str,
    X: np.ndarray,
    y: np.ndarray,
    classifier: str,
) -> Dict:
    """扩展 8 子带 FBCSP + LDA / SVM。"""
    baseline = BASELINE_CV[subject]
    bands_str = "+".join(f"{a}-{b}" for a, b in EXTENDED_BANDS)

    def factory():
        return ExtendedFBCSP(
            bands=EXTENDED_BANDS,
            n_components=2,
            classifier=classifier,
        )

    metrics = leakage_safe_cv_evaluate(X, y, factory, random_state=RANDOM_STATE)
    clf_name = "LDA" if classifier == "lda" else "SVM"
    return _result_row(
        subject,
        "extended_fbcsp",
        f"bands=[{bands_str}] n_csp=2->{clf_name}",
        metrics,
        baseline,
    )


def run_csp_dim_search(subject: str, X: np.ndarray, y: np.ndarray) -> List[Dict]:
    """CSP n_components 搜索 [2,4,6,8] + SVM。"""
    baseline = BASELINE_CV[subject]
    rows = []
    for n in CSP_DIMS:
        def factory(n=n):
            return Pipeline([
                ("csp", CSP(n_components=n, reg="ledoit_wolf", log=True)),
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel="rbf", random_state=RANDOM_STATE)),
            ])

        metrics = leakage_safe_cv_evaluate(X, y, factory, random_state=RANDOM_STATE)
        rows.append(_result_row(subject, "csp_dim_search", f"n_components={n}", metrics, baseline))
    return rows


def run_svm_comparison(subject: str, X: np.ndarray, y: np.ndarray) -> List[Dict]:
    """RBF SVM 网格 + 线性 SVM 基线（CSP 6 维特征）。"""
    baseline = BASELINE_CV[subject]
    rows = []

    def linear_factory():
        return Pipeline([
            ("csp", CSP(n_components=6, reg="ledoit_wolf", log=True)),
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="linear", random_state=RANDOM_STATE)),
        ])

    metrics = leakage_safe_cv_evaluate(X, y, linear_factory, random_state=RANDOM_STATE)
    rows.append(_result_row(subject, "svm_linear", "kernel=linear n_csp=6", metrics, baseline))

    for c in SVM_C_VALUES:
        for gamma in SVM_GAMMA_VALUES:
            def factory(c=c, gamma=gamma):
                return Pipeline([
                    ("csp", CSP(n_components=6, reg="ledoit_wolf", log=True)),
                    ("scaler", StandardScaler()),
                    ("svm", SVC(kernel="rbf", C=c, gamma=gamma, random_state=RANDOM_STATE)),
                ])

            metrics = leakage_safe_cv_evaluate(X, y, factory, random_state=RANDOM_STATE)
            rows.append(
                _result_row(
                    subject,
                    "svm_rbf",
                    f"kernel=rbf C={c} gamma={gamma} n_csp=6",
                    metrics,
                    baseline,
                )
            )
    return rows
