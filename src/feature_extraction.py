"""
EEG 特征提取模块
- 带通滤波
- 对数频带功率（μ / β / μ+β）
- 可选 CSP 空间滤波
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.signal import butter, filtfilt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

from config import FREQUENCY_BANDS, MI_TIME_MIN, SAMPLE_RATE


def get_eeg_columns(df: pd.DataFrame) -> List[str]:
    """获取 EEG 通道列名。"""
    return [col for col in df.columns if col.startswith("EEG")]


def bandpass_filter(signal: np.ndarray, sample_rate: float, fmin: float, fmax: float) -> np.ndarray:
    """对多通道 EEG 信号做带通滤波。signal shape: (n_channels, n_samples)"""
    nyquist = sample_rate / 2.0
    low = fmin / nyquist
    high = fmax / nyquist
    if low <= 0 or high >= 1:
        raise ValueError(f"频带 [{fmin}, {fmax}] Hz 超出采样率 {sample_rate} Hz 的有效范围")
    b, a = butter(4, [low, high], btype="band")
    return filtfilt(b, a, signal, axis=1)


def regularize_cov(cov: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """对协方差矩阵做正则化，避免奇异矩阵导致 CSP 失败。"""
    return cov + eps * np.eye(cov.shape[0])


def log_bandpower(signal: np.ndarray) -> np.ndarray:
    """计算各通道对数功率。"""
    return np.log(np.var(signal, axis=1) + 1e-8)


class EEGFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    从原始 EEG 时序 CSV 提取试次级特征。

    每个 (patient, epoch) 输出一个特征向量：
    - 默认：3 个频带 × 22 通道 = 66 维对数频带功率
    - csp_only=True：经典 CSP 对数方差特征（配合 SVM 使用）
    - use_csp=True：在频带功率基础上追加 CSP 特征
    """

    def __init__(
        self,
        sample_rate: float = SAMPLE_RATE,
        time_min: float = MI_TIME_MIN,
        frequency_bands: Optional[List[Tuple[str, float, float]]] = None,
        use_csp: bool = False,
        csp_only: bool = False,
        n_csp_components: int = 2,
    ):
        self.sample_rate = sample_rate
        self.time_min = time_min
        self.frequency_bands = frequency_bands or FREQUENCY_BANDS
        self.use_csp = use_csp or csp_only
        self.csp_only = csp_only
        self.n_csp_components = n_csp_components

        self.eeg_columns_: List[str] = []
        self.csp_filters_: Optional[np.ndarray] = None
        self.feature_names_: List[str] = []

    def _iter_epochs(self, df: pd.DataFrame):
        """按试次遍历 EEG 片段。"""
        eeg_cols = self.eeg_columns_ or get_eeg_columns(df)
        filtered = df[df["time"] >= self.time_min]

        for (patient, epoch), group in filtered.groupby(["patient", "epoch"]):
            ordered = group.sort_values("time")
            signal = ordered[eeg_cols].values.T
            if signal.shape[1] < 10:
                continue
            label = ordered["label"].iloc[0] if "label" in ordered.columns else None
            yield patient, epoch, signal, label

    def _bandpower_features(self, signal: np.ndarray) -> np.ndarray:
        """提取多频带对数功率特征。"""
        features = []
        for _, fmin, fmax in self.frequency_bands:
            filtered = bandpass_filter(signal, self.sample_rate, fmin, fmax)
            features.extend(log_bandpower(filtered))
        return np.array(features)

    def _fit_csp(self, trials: List[np.ndarray], labels: np.ndarray) -> np.ndarray:
        """训练 CSP 滤波器（one-vs-rest）。"""
        n_channels = trials[0].shape[0]
        n_classes = len(np.unique(labels))
        n_comp = min(self.n_csp_components, n_channels // 2)
        filters = []

        for class_id in range(n_classes):
            class_trials = [trials[i] for i in range(len(trials)) if labels[i] == class_id]
            rest_trials = [trials[i] for i in range(len(trials)) if labels[i] != class_id]

            def average_normalized_cov(trial_list: List[np.ndarray]) -> np.ndarray:
                cov_sum = np.zeros((n_channels, n_channels))
                for trial in trial_list:
                    cov = np.cov(trial)
                    trace = np.trace(cov)
                    if trace <= 0:
                        trace = 1e-8
                    cov_sum += cov / trace
                return regularize_cov(cov_sum / len(trial_list))

            cov_class = average_normalized_cov(class_trials)
            cov_rest = average_normalized_cov(rest_trials)
            composite = regularize_cov(cov_class + cov_rest)
            eigvals, eigvecs = eigh(cov_class, composite)
            order = np.argsort(eigvals)[::-1]
            filters.append(eigvecs[:, order[:n_comp]].T)

        return np.vstack(filters)

    def _csp_features(self, signal: np.ndarray) -> np.ndarray:
        """用已训练的 CSP 滤波器提取特征。"""
        if self.csp_filters_ is None:
            return np.array([])

        projected = self.csp_filters_ @ signal
        variance = np.var(projected, axis=1)
        total = np.sum(variance) + 1e-8
        return np.log(variance / total + 1e-8)

    def _build_feature_names(self, n_channels: int) -> List[str]:
        """生成可读的特征名列表。"""
        names = []
        if not self.csp_only:
            eeg_cols = self.eeg_columns_ or [f"ch{i}" for i in range(n_channels)]
            for band_name, _, _ in self.frequency_bands:
                for col in eeg_cols:
                    names.append(f"{band_name}_{col}")
        if self.use_csp and self.csp_filters_ is not None:
            for i in range(self.csp_filters_.shape[0]):
                names.append(f"csp_{i}")
        return names

    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None):
        """拟合 CSP（若启用）。"""
        self.eeg_columns_ = get_eeg_columns(df)

        if not self.use_csp:
            self.feature_names_ = self._build_feature_names(len(self.eeg_columns_))
            return self

        trials = []
        labels = []
        for _, _, signal, label in self._iter_epochs(df):
            # CSP 使用 μ+β 宽带信号
            filtered = bandpass_filter(signal, self.sample_rate, 8, 30)
            trials.append(filtered)
            labels.append(label)

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(labels)
        self.csp_filters_ = self._fit_csp(trials, y_encoded)
        self.feature_names_ = self._build_feature_names(len(self.eeg_columns_))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """将原始 CSV 转为特征矩阵。"""
        if not self.eeg_columns_:
            self.eeg_columns_ = get_eeg_columns(df)

        rows = []
        for _, _, signal, _ in self._iter_epochs(df):
            if self.csp_only:
                if self.csp_filters_ is None:
                    raise ValueError("CSP 模式需要先调用 fit() 训练 CSP 滤波器。")
                filtered = bandpass_filter(signal, self.sample_rate, 8, 30)
                rows.append(self._csp_features(filtered))
            elif self.use_csp and self.csp_filters_ is not None:
                band_features = self._bandpower_features(signal)
                filtered = bandpass_filter(signal, self.sample_rate, 8, 30)
                csp_features = self._csp_features(filtered)
                rows.append(np.concatenate([band_features, csp_features]))
            else:
                rows.append(self._bandpower_features(signal))

        if not rows:
            raise ValueError("未能从数据中提取任何试次特征，请检查 CSV 格式。")

        return np.vstack(rows)

    def extract_labels(self, df: pd.DataFrame) -> np.ndarray:
        """提取与 transform 输出对齐的标签。"""
        labels = []
        for _, _, _, label in self._iter_epochs(df):
            labels.append(label)
        return np.array(labels)

    def extract_groups(self, df: pd.DataFrame) -> np.ndarray:
        """提取与 transform 输出对齐的 patient 分组。"""
        groups = []
        for patient, _, _, _ in self._iter_epochs(df):
            groups.append(patient)
        return np.array(groups)


def extract_features_from_raw(
    df: pd.DataFrame,
    use_csp: bool = False,
    fit_csp: bool = True,
    extractor: Optional[EEGFeatureExtractor] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, EEGFeatureExtractor]:
    """
    从原始 EEG CSV 提取特征矩阵、标签和 patient 分组。

    返回:
        X, y, groups, extractor
    """
    if extractor is None:
        extractor = EEGFeatureExtractor(use_csp=use_csp)

    if use_csp and fit_csp:
        extractor.fit(df)

    X = extractor.transform(df)
    y = extractor.extract_labels(df)
    groups = extractor.extract_groups(df)
    return X, y, groups, extractor
