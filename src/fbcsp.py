"""
Filter Bank CSP + LDA（经典 BCI baseline）
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from mne.decoding import CSP
from scipy.signal import butter, filtfilt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler


def _subbands_for_range(fmin: float, fmax: float, width: float = 4.0) -> List[Tuple[float, float]]:
    """将宽频带拆成 4Hz 子带。"""
    bands = []
    start = fmin
    while start + width <= fmax:
        bands.append((start, start + width))
        start += width
    if not bands:
        bands = [(fmin, fmax)]
    return bands


class FBCSP_LDA(BaseEstimator, ClassifierMixin):
    """
    Filter Bank CSP + LDA。
    在 fit 阶段对每个子带独立训练 CSP，拼接 log-variance 特征后 LDA 分类。
    """

    def __init__(
        self,
        bandpass: Tuple[float, float] = (8.0, 30.0),
        n_components: int = 2,
        sample_rate: float = 250.0,
        subband_width: float = 8.0,
    ):
        self.bandpass = bandpass
        self.n_components = n_components
        self.sample_rate = sample_rate
        self.subband_width = subband_width

    def _filter_band(self, X: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
        nyq = self.sample_rate / 2.0
        b, a = butter(4, [fmin / nyq, fmax / nyq], btype="band")
        return filtfilt(b, a, X, axis=-1)

    def _extract_features(self, X: np.ndarray, y=None, fit: bool = False) -> np.ndarray:
        bands = _subbands_for_range(self.bandpass[0], self.bandpass[1], self.subband_width)
        if fit:
            self.csps_ = []

        feats = []
        for i, (fmin, fmax) in enumerate(bands):
            X_band = self._filter_band(X, fmin, fmax)
            if fit:
                csp = CSP(n_components=self.n_components, reg="ledoit_wolf", log=True)
                csp.fit(X_band, y)
                self.csps_.append(csp)
            feats.append(self.csps_[i].transform(X_band))

        return np.hstack(feats)

    def fit(self, X, y):
        X = np.asarray(X)
        features = self._extract_features(X, y, fit=True)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(features)
        self.lda_ = LinearDiscriminantAnalysis()
        self.lda_.fit(X_scaled, y)
        return self

    def predict(self, X):
        features = self._extract_features(np.asarray(X), fit=False)
        return self.lda_.predict(self.scaler_.transform(features))

    def score(self, X, y):
        return float((self.predict(X) == y).mean())
