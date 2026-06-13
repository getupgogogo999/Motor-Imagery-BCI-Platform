"""
数据预处理模块
- 缺失值检查与处理
- 标签编码
- 特征标准化
- 支持按 patient 分组划分
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import RANDOM_STATE, TEST_SIZE


def check_missing_values(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """检查并报告缺失值情况。"""
    missing = df[feature_cols].isnull().sum()
    total_missing = missing.sum()
    print("\n========== 缺失值检查 ==========")
    if total_missing == 0:
        print("未发现缺失值。")
    else:
        print(f"共发现 {total_missing} 个缺失值:")
        print(missing[missing > 0])
    print("================================\n")


def handle_missing_values(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """用列均值填充缺失值（若存在）。"""
    df = df.copy()
    if df[feature_cols].isnull().any().any():
        print("正在用均值填充缺失值...")
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())
    return df


def encode_labels(y: Union[np.ndarray, pd.Series]) -> Tuple[np.ndarray, LabelEncoder]:
    """将文字标签编码为数字。"""
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    print(f"标签编码映射: {dict(zip(encoder.classes_, range(len(encoder.classes_))))}")
    return y_encoded, encoder


def normalize_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """对特征进行标准化（零均值、单位方差）。"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("特征标准化完成。")
    return X_train_scaled, X_test_scaled, scaler


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    split_by: str = "epoch",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    划分训练集和测试集。

    split_by:
        - epoch: 按试次随机分层划分（单患者训练推荐）
        - patient: 按 patient 分组划分（多患者泛化评估）
    """
    if split_by == "patient" and groups is not None and len(set(groups)) > 1:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
        train_idx, test_idx = next(splitter.split(X, y, groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_patients = sorted(set(groups[train_idx]))
        test_patients = sorted(set(groups[test_idx]))
        print(f"按 patient 划分: 训练 {train_patients}, 测试 {test_patients}")
        print(f"训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")
        return X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"按 epoch 随机划分: 训练集 {len(X_train)} 样本, 测试集 {len(X_test)} 样本")
    return X_train, X_test, y_train, y_test


def split_raw_dataframe(
    df: pd.DataFrame,
    label_column: str,
    split_by: str = "epoch",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """在特征提取前划分原始时序数据，避免 CSP 等信息泄漏。"""
    meta = df.groupby(["patient", "epoch"], as_index=False).agg({label_column: "first"})
    groups = meta["patient"].values
    y_raw = meta[label_column].values

    if split_by == "patient" and len(set(groups)) > 1:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
        train_idx, test_idx = next(splitter.split(meta, y_raw, groups))
        train_patients = sorted(meta.iloc[train_idx]["patient"].unique())
        test_patients = sorted(meta.iloc[test_idx]["patient"].unique())
        print(f"按 patient 划分: 训练 {train_patients}, 测试 {test_patients}")
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(meta)),
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_raw,
        )
        print("按 epoch 随机划分训练/测试试次")

    train_meta = meta.iloc[train_idx][["patient", "epoch"]]
    test_meta = meta.iloc[test_idx][["patient", "epoch"]]

    df_train = df.merge(train_meta, on=["patient", "epoch"], how="inner")
    df_test = df.merge(test_meta, on=["patient", "epoch"], how="inner")
    print(f"训练试次: {len(train_meta)}, 测试试次: {len(test_meta)}")
    return df_train, df_test
