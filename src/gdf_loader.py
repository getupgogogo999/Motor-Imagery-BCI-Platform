"""
GDF 数据加载模块（BCI Competition IV Dataset 2A）
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import mne
import numpy as np

from config import (
    GDF_BANDPASS,
    GDF_DIR,
    GDF_EPOCH_TMAX,
    GDF_EPOCH_TMIN,
    MI_EVENT_CODES,
)

mne.set_log_level("ERROR")


def list_gdf_subjects(gdf_dir: Path | None = None, session: str = "T") -> List[str]:
    """列出可用的受试者 ID，如 A01, A02, ..."""
    folder = gdf_dir or GDF_DIR
    suffix = f"{session.upper()}.gdf"
    subjects = sorted({p.stem[:3] for p in folder.glob("A*T.gdf")})
    return subjects


def subject_to_gdf_path(subject: str, session: str = "T", gdf_dir: Path | None = None) -> Path:
    """将 1 / A01 / a01 转为 GDF 文件路径。"""
    folder = gdf_dir or GDF_DIR
    subject = subject.upper().strip()
    if subject.isdigit():
        subject = f"A{int(subject):02d}"
    if not subject.startswith("A"):
        subject = f"A{subject}"
    if len(subject) == 2 and subject[1].isdigit():
        subject = f"A0{subject[1]}"
    filename = f"{subject[:3]}{session.upper()}.gdf"
    path = folder / filename
    if not path.exists():
        raise FileNotFoundError(f"未找到 GDF 文件: {path}")
    return path


def load_mi_epochs(
    gdf_path: Path | str,
    tmin: float = GDF_EPOCH_TMIN,
    tmax: float = GDF_EPOCH_TMAX,
    bandpass: Tuple[float, float] = GDF_BANDPASS,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    从 GDF 加载运动想象 epoch。

    返回:
        X: (n_trials, n_channels, n_times)
        y: (n_trials,) 整数标签 0..3
        class_names: 如 ['foot', 'left', 'right', 'tongue']
    """
    raw = mne.io.read_raw_gdf(str(gdf_path), preload=True)
    raw.pick("eeg")
    raw.filter(bandpass[0], bandpass[1], verbose=False)

    events, event_id = mne.events_from_annotations(raw)
    selected_event_id = {
        str(code): event_id[str(code)]
        for code in MI_EVENT_CODES
        if str(code) in event_id
    }
    if len(selected_event_id) < 4:
        raise ValueError(f"GDF 中未找到完整的 4 类 MI 事件: {gdf_path}")

    epochs = mne.Epochs(
        raw,
        events,
        event_id=selected_event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False,
    )

    code_to_name = {
        event_id[str(code)]: name for code, name in MI_EVENT_CODES.items()
    }
    y_names = [code_to_name[event] for event in epochs.events[:, 2]]
    class_names = ["foot", "left", "right", "tongue"]
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    y = np.array([name_to_idx[name] for name in y_names], dtype=int)
    X = epochs.get_data()

    print(
        f"已加载 {Path(gdf_path).name}: {len(y)} 试次, "
        f"{X.shape[1]} 通道, {X.shape[2]} 时间点/试次"
    )
    print(f"窗口: {tmin}~{tmax}s, 滤波: {bandpass[0]}-{bandpass[1]} Hz")
    return X, y, class_names
