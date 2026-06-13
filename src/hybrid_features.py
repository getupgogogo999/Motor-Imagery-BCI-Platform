"""
CSP (FBCSP) + Riemannian Tangent Space 特征融合
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from src.paper_fbcsp import PaperFBCSP


class HybridCSPRiemannian(BaseEstimator, ClassifierMixin):
    """
    FBCSP log-var 特征 + Riemannian 切空间特征拼接 → StandardScaler → LDA。
    """

    def __init__(
        self,
        top_k_fbcsp: int = 22,
        sample_rate: float = 250.0,
    ):
        self.top_k_fbcsp = top_k_fbcsp
        self.sample_rate = sample_rate

    def _fbcsp_features(self, X: np.ndarray, y=None, fit: bool = False) -> np.ndarray:
        if fit:
            self.fbcsp_ = PaperFBCSP(top_k_features=self.top_k_fbcsp, sample_rate=self.sample_rate)
            self.fbcsp_.fit(X, y)
        return self.fbcsp_.transform_features(X)

    def _riemann_features(self, X: np.ndarray, y=None, fit: bool = False) -> np.ndarray:
        if fit:
            self.cov_ = Covariances(estimator="lwf")
            self.ts_ = TangentSpace()
            cov = self.cov_.fit_transform(X)
            self.ts_.fit(cov)
        cov = self.cov_.transform(X)
        return self.ts_.transform(cov)

    def _fuse(self, X: np.ndarray, y=None, fit: bool = False) -> np.ndarray:
        fbcsp = self._fbcsp_features(X, y, fit)
        riem = self._riemann_features(X, y, fit)
        return np.hstack([fbcsp, riem])

    def fit(self, X, y):
        X = np.asarray(X)
        fused = self._fuse(X, y, fit=True)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(fused)
        self.clf_ = LinearDiscriminantAnalysis()
        self.clf_.fit(X_scaled, y)
        return self

    def predict(self, X):
        fused = self._fuse(np.asarray(X), fit=False)
        return self.clf_.predict(self.scaler_.transform(fused))

    def score(self, X, y):
        return float((self.predict(X) == y).mean())


class FeatureFusionExtractor(BaseEstimator):
    """仅提取融合特征，供多分类器搜索。"""

    def __init__(self, top_k_fbcsp: int = 22, sample_rate: float = 250.0):
        self.top_k_fbcsp = top_k_fbcsp
        self.sample_rate = sample_rate

    def fit(self, X, y):
        self.hybrid_ = HybridCSPRiemannian(top_k_fbcsp=self.top_k_fbcsp, sample_rate=self.sample_rate)
        self.hybrid_.fit(X, y)
        return self

    def transform(self, X):
        return self.hybrid_._fuse(np.asarray(X), fit=False)

    def fit_transform(self, X, y):
        self.fit(X, y)
        fused = self.hybrid_._fuse(np.asarray(X), fit=False)
        return self.hybrid_.scaler_.transform(fused)
