"""
受试者级数据诊断：试次数、类别、坏通道、频带能量等
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from config import GDF_DIR, OUTPUTS_DIR
from src.gdf_loader import subject_to_gdf_path
from src.gdf_preprocessing import CLASS_NAMES, load_unfiltered_epochs


def bandpower_per_channel(data: np.ndarray, sfreq: float, fmin: float, fmax: float) -> np.ndarray:
    """每个通道的平均对数 bandpower。data: (trials, ch, time)"""
    nyq = sfreq / 2.0
    b, a = butter(4, [fmin / nyq, fmax / nyq], btype="band")
    filtered = filtfilt(b, a, data, axis=-1)
    return np.log(np.var(filtered, axis=-1).mean(axis=0) + 1e-8)


def detect_bad_channels(data: np.ndarray, z_threshold: float = 3.0) -> List[int]:
    """
    基于通道方差的 Z 分数检测坏通道。
    返回坏通道索引列表。
    """
    var_per_ch = np.var(data, axis=(0, 2))
    med = np.median(var_per_ch)
    mad = np.median(np.abs(var_per_ch - med)) + 1e-8
    z = 0.6745 * (var_per_ch - med) / mad
    return np.where(np.abs(z) > z_threshold)[0].tolist()


def diagnose_subject(subject: str, gdf_dir: Path | None = None) -> Dict:
    """计算单个受试者的详细统计。"""
    path = subject_to_gdf_path(subject, session="T", gdf_dir=gdf_dir or GDF_DIR)
    X, y, sfreq, ch_names = load_unfiltered_epochs(path)

    bad_idx = detect_bad_channels(X)
    mu_power = bandpower_per_channel(X, sfreq, 8, 13)
    beta_power = bandpower_per_channel(X, sfreq, 13, 30)

    class_counts = {CLASS_NAMES[i]: int((y == i).sum()) for i in range(len(CLASS_NAMES))}

    row = {
        "subject": subject[:3] if subject.startswith("A") else f"A{int(subject):02d}",
        "n_trials": len(y),
        "n_channels": X.shape[1],
        "n_times": X.shape[2],
        "bad_channel_count": len(bad_idx),
        "bad_channels": ",".join([ch_names[i] for i in bad_idx]) if bad_idx else "",
        "eeg_amplitude_mean": float(np.mean(X)),
        "eeg_amplitude_std": float(np.std(X)),
        "mu_power_mean": float(np.mean(mu_power)),
        "mu_power_std": float(np.std(mu_power)),
        "beta_power_mean": float(np.mean(beta_power)),
        "beta_power_std": float(np.std(beta_power)),
        "class_foot": class_counts["foot"],
        "class_left": class_counts["left"],
        "class_right": class_counts["right"],
        "class_tongue": class_counts["tongue"],
        "class_imbalance_max_min_ratio": max(class_counts.values()) / max(min(class_counts.values()), 1),
    }
    return row


def run_diagnostics(gdf_dir: Path | None = None) -> pd.DataFrame:
    """诊断全部受试者并保存 CSV。"""
    rows = []
    for i in range(1, 10):
        subject = f"A{i:02d}"
        try:
            rows.append(diagnose_subject(subject, gdf_dir))
        except FileNotFoundError:
            continue

    df = pd.DataFrame(rows)
    out_dir = OUTPUTS_DIR / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "subject_diagnostics.csv", index=False)

    # 标记异常受试者：坏通道多、μ/β 能量极端、或振幅异常
    if not df.empty:
        df["anomaly_flag"] = (
            (df["bad_channel_count"] >= 2)
            | (df["mu_power_mean"] < df["mu_power_mean"].quantile(0.1))
            | (df["mu_power_mean"] > df["mu_power_mean"].quantile(0.9))
            | (df["eeg_amplitude_std"] > df["eeg_amplitude_std"].quantile(0.9))
        )
        df.to_csv(out_dir / "subject_diagnostics.csv", index=False)

    return df


def print_diagnostics_summary(df: pd.DataFrame) -> None:
    """打印诊断摘要。"""
    print("\n========== 受试者诊断摘要 ==========")
    cols = [
        "subject", "n_trials", "bad_channel_count", "eeg_amplitude_std",
        "mu_power_mean", "beta_power_mean", "anomaly_flag",
    ]
    if "anomaly_flag" in df.columns:
        print(df[cols].to_string(index=False))
        flagged = df[df["anomaly_flag"]]["subject"].tolist()
        if flagged:
            print(f"\n疑似异常受试者: {flagged}")
        else:
            print("\n未发现明显异常受试者（基于启发式规则）")
    print("====================================\n")
