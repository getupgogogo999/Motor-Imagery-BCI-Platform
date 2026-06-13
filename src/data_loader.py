"""
数据加载模块
- 读取 CSV / 分患者文件夹
- 自动检查数据结构
- 自动识别或使用指定的标签列
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from config import (
    DEFAULT_DATA_PATH,
    EXCLUDED_PATIENT_IDS,
    LABEL_COLUMN_CANDIDATES,
    PATIENTS_DIR,
)


def load_csv(file_path: Optional[str] = None) -> pd.DataFrame:
    """读取单个 CSV 文件并返回 DataFrame。"""
    path = file_path or str(DEFAULT_DATA_PATH)
    print(f"正在读取数据: {path}")
    df = pd.read_csv(path)
    print(f"读取完成，共 {len(df):,} 行，{len(df.columns)} 列")
    return df


def load_patient_file(patient_id: int, patients_dir: Optional[Path] = None) -> pd.DataFrame:
    """读取指定患者的 CSV 文件。"""
    folder = patients_dir or PATIENTS_DIR
    file_path = folder / f"BCICIV_2a_{patient_id}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"未找到患者数据文件: {file_path}")
    return load_csv(str(file_path))


def load_all_patients(
    patients_dir: Optional[Path] = None,
    exclude_ids: Optional[set] = None,
) -> pd.DataFrame:
    """读取 patients 文件夹中所有完整（4 类）患者数据并合并。"""
    folder = patients_dir or PATIENTS_DIR
    skip = exclude_ids if exclude_ids is not None else EXCLUDED_PATIENT_IDS

    if not folder.exists():
        raise FileNotFoundError(f"分患者数据目录不存在: {folder}")

    csv_files = sorted(folder.glob("BCICIV_2a_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"目录中未找到 BCICIV_2a_*.csv 文件: {folder}")

    frames: List[pd.DataFrame] = []
    print(f"正在从 {folder} 加载分患者数据...")

    for file_path in csv_files:
        patient_id = int(file_path.stem.split("_")[-1])
        if patient_id in skip:
            print(f"  跳过 patient {patient_id}（仅含部分类别）")
            continue

        df = pd.read_csv(file_path)
        frames.append(df)
        print(f"  已加载 patient {patient_id}: {len(df):,} 行")

    if not frames:
        raise ValueError("没有可用的患者数据文件。")

    combined = pd.concat(frames, ignore_index=True)
    print(f"合并完成，共 {len(combined):,} 行，{len(frames)} 名患者")
    return combined


def inspect_data(df: pd.DataFrame) -> None:
    """打印数据基本信息，帮助了解数据结构。"""
    print("\n========== 数据结构概览 ==========")
    print(f"行数: {len(df):,}")
    print(f"列数: {len(df.columns)}")

    if "patient" in df.columns:
        print(f"患者数: {df['patient'].nunique()}")
        print(f"患者列表: {sorted(df['patient'].unique())}")

    if "time" in df.columns:
        print(f"时间范围: {df['time'].min()} ~ {df['time'].max()} 秒")

    if "label" in df.columns and "epoch" in df.columns:
        print(f"各类别试次数:\n{df.groupby('label')['epoch'].nunique()}")

    print("\n前 5 列名:", list(df.columns[:5]))
    print("EEG 通道数:", len([c for c in df.columns if c.startswith("EEG")]))
    print("==================================\n")


def detect_label_column(
    df: pd.DataFrame,
    label_column: Optional[str] = None,
) -> str:
    """自动识别标签列，或由用户指定。"""
    if label_column:
        if label_column not in df.columns:
            raise ValueError(f"指定的标签列 '{label_column}' 不存在于数据中")
        print(f"使用用户指定的标签列: {label_column}")
        return label_column

    for candidate in LABEL_COLUMN_CANDIDATES:
        if candidate in df.columns:
            print(f"自动识别标签列: {candidate}")
            return candidate

    raise ValueError(
        "无法自动识别标签列，请通过 label_column 参数手动指定。"
        f" 当前列名: {list(df.columns)}"
    )


def get_feature_columns(df: pd.DataFrame, label_column: str) -> list[str]:
    """获取 EEG 特征列。"""
    feature_cols = [col for col in df.columns if col.startswith("EEG")]
    if not feature_cols:
        meta_cols = {"patient", "time", "epoch", label_column}
        feature_cols = [
            col
            for col in df.select_dtypes(include="number").columns
            if col not in meta_cols
        ]
    print(f"共找到 {len(feature_cols)} 个 EEG 特征列")
    return feature_cols


def aggregate_by_epoch(df: pd.DataFrame, feature_cols: list[str], label_column: str) -> pd.DataFrame:
    """按 (patient, epoch) 聚合（旧版简单均值特征，保留用于兼容）。"""
    print("正在按 epoch 聚合数据（每个试次取 EEG 均值）...")
    grouped = (
        df.groupby(["patient", "epoch"], as_index=False)
        .agg({label_column: "first", **{col: "mean" for col in feature_cols}})
    )
    print(f"聚合后样本数: {len(grouped):,}")
    print(f"各类别分布:\n{grouped[label_column].value_counts()}")
    return grouped
