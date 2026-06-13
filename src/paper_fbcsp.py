"""
论文版 FBCSP (Ang et al., 2008)

- 8 个 4Hz 子带 Filter Bank
- 每子带 m=2 CSP，log-variance 特征
- Mutual Information 特征选择
- LDA 分类
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from mne.decoding import CSP
from scipy.signal import butter, filtfilt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler

# 8 子带 × 4Hz（4–36 Hz）
PAPER_FBCSP_BANDS: List[Tuple[float, float]] = [
    (4, 8), (8, 12), (12, 16), (16, 20),
    (20, 24), (24, 28), (28, 32), (32, 36),
]


class PaperFBCSP(BaseEstimator, ClassifierMixin):
    """
    论文标准 FBCSP 流程。
    n_components_per_band=2 → 16 维特征 → MI 选 top_k → LDA。
    """

    def __init__(
        self,
        bands: Optional[List[Tuple[float, float]]] = None,
        n_components_per_band: int = 2,
        top_k_features: int = 22,
        sample_rate: float = 250.0,
    ):
        self.bands = bands or PAPER_FBCSP_BANDS
        self.n_components_per_band = n_components_per_band
        self.top_k_features = top_k_features
        self.sample_rate = sample_rate

    def _filter_band(self, X: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
        nyq = self.sample_rate / 2.0
        low, high = max(fmin / nyq, 1e-4), min(fmax / nyq, 0.999)
        b, a = butter(4, [low, high], btype="band")
        return filtfilt(b, a, X, axis=-1)

    def _extract_fbcsp_features(self, X: np.ndarray, y=None, fit: bool = False) -> np.ndarray:
        if fit:
            self.csps_ = []
        feats = []
        for i, (fmin, fmax) in enumerate(self.bands):
            Xb = self._filter_band(X, fmin, fmax)
            if fit:
                csp = CSP(
                    n_components=self.n_components_per_band,
                    reg="ledoit_wolf",
                    log=True,
                    norm_trace=False,
                )
                csp.fit(Xb, y)
                self.csps_.append(csp)
            feats.append(self.csps_[i].transform(Xb))
        return np.hstack(feats)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        features = self._extract_fbcsp_features(X, y, fit=True)
        n_feat = features.shape[1]
        k = min(self.top_k_features, n_feat)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(features)
        self.selector_ = SelectKBest(mutual_info_classif, k=k)
        X_sel = self.selector_.fit_transform(X_scaled, y)
        self.lda_ = LinearDiscriminantAnalysis()
        self.lda_.fit(X_sel, y)
        return self

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """返回 MI 选择后的 FBCSP 特征（供融合/分类器搜索）。"""
        features = self._extract_fbcsp_features(np.asarray(X), fit=False)
        X_scaled = self.scaler_.transform(features)
        return self.selector_.transform(X_scaled)

    def predict(self, X):
        return self.lda_.predict(self.transform_features(X))

    def score(self, X, y):
        return float((self.predict(X) == y).mean())
