"""
GDF 预处理：CAR、可配置滤波与 epoch 提取
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import mne
import numpy as np
from scipy.signal import butter, filtfilt

from config import GDF_DIR, MI_EVENT_CODES

mne.set_log_level("ERROR")

CLASS_NAMES = ["foot", "left", "right", "tongue"]
CLASS_NAMES_LR = ["left", "right"]
LR_EVENT_CODES = {769: "left", 770: "right"}


def apply_car(data: np.ndarray) -> np.ndarray:
    """
    Common Average Reference：每个时间点减去所有通道均值。
    data shape: (n_trials, n_channels, n_times)
    """
    return data - data.mean(axis=1, keepdims=True)


def bandpass_trials(
    data: np.ndarray,
    sample_rate: float,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """对试次数据做带通滤波。"""
    nyq = sample_rate / 2.0
    b, a = butter(4, [fmin / nyq, fmax / nyq], btype="band")
    return filtfilt(b, a, data, axis=-1)


def load_mi_epochs_flexible(
    gdf_path: Path | str,
    tmin: float = 0.5,
    tmax: float = 2.5,
    bandpass: Optional[Tuple[float, float]] = None,
    apply_car_ref: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str], float, List[str]]:
    """
    加载 MI epoch，支持 CAR 与可选滤波。

    返回:
        X, y, class_names, sample_rate, channel_names
    """
    raw = mne.io.read_raw_gdf(str(gdf_path), preload=True)
    raw.pick("eeg")
    sfreq = float(raw.info["sfreq"])
    ch_names = list(raw.ch_names)

    events, event_id = mne.events_from_annotations(raw)
    selected_event_id = {
        str(code): event_id[str(code)]
        for code in MI_EVENT_CODES
        if str(code) in event_id
    }
    if len(selected_event_id) < 4:
        raise ValueError(f"缺少完整 4 类 MI 事件: {gdf_path}")

    if bandpass is not None:
        raw.filter(bandpass[0], bandpass[1], verbose=False)

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

    code_to_name = {event_id[str(code)]: name for code, name in MI_EVENT_CODES.items()}
    y_names = [code_to_name[e] for e in epochs.events[:, 2]]
    name_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    y = np.array([name_to_idx[name] for name in y_names], dtype=int)
    X = epochs.get_data()

    if apply_car_ref:
        X = apply_car(X)

    if verbose:
        print(
            f"{Path(gdf_path).name}: trials={len(y)}, ch={X.shape[1]}, "
            f"times={X.shape[2]}, window={tmin}~{tmax}s, "
            f"band={bandpass}, CAR={apply_car_ref}"
        )

    return X, y, CLASS_NAMES, sfreq, ch_names


def load_mi_epochs_lr_flexible(
    gdf_path: Path | str,
    tmin: float = 0.5,
    tmax: float = 2.5,
    bandpass: Optional[Tuple[float, float]] = None,
    apply_car_ref: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str], float, List[str]]:
    """
    加载左右手二分类 MI epoch（BCI Competition IV 2b 等）。
    事件码 769=左手, 770=右手。
    """
    raw = mne.io.read_raw_gdf(str(gdf_path), preload=True)
    eeg_names = [c for c in raw.ch_names if c.upper().startswith("EEG:") and "EOG" not in c.upper()]
    if len(eeg_names) >= 3:
        raw.pick(eeg_names[:3])
    else:
        raw.pick("eeg")
    sfreq = float(raw.info["sfreq"])
    ch_names = list(raw.ch_names)

    events, event_id = mne.events_from_annotations(raw)
    selected_event_id = {
        str(code): event_id[str(code)]
        for code in LR_EVENT_CODES
        if str(code) in event_id
    }
    if len(selected_event_id) < 2:
        raise ValueError(f"缺少左右手 MI 事件 (769/770): {gdf_path}")

    if bandpass is not None:
        raw.filter(bandpass[0], bandpass[1], verbose=False)

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

    code_to_name = {event_id[str(code)]: name for code, name in LR_EVENT_CODES.items()}
    y_names = [code_to_name[e] for e in epochs.events[:, 2]]
    name_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES_LR)}
    y = np.array([name_to_idx[name] for name in y_names], dtype=int)
    X = epochs.get_data()

    if apply_car_ref:
        X = apply_car(X)

    if verbose:
        print(
            f"{Path(gdf_path).name} [LR]: trials={len(y)}, ch={X.shape[1]}, "
            f"times={X.shape[2]}, window={tmin}~{tmax}s"
        )

    return X, y, CLASS_NAMES_LR, sfreq, ch_names


def load_epochs_auto(
    gdf_path: Path | str,
    gdf_format: str = "auto",
    tmin: float = 0.5,
    tmax: float = 2.5,
    bandpass: Optional[Tuple[float, float]] = None,
    apply_car_ref: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str], float, List[str]]:
    """按 gdf_format 或文件内容自动选择 4 类 / 左右手 2 类加载。"""
    path = Path(gdf_path)
    if gdf_format == "bci2b_lr":
        return load_mi_epochs_lr_flexible(
            path, tmin=tmin, tmax=tmax, bandpass=bandpass,
            apply_car_ref=apply_car_ref, verbose=verbose,
        )
    if gdf_format == "bci2a_4class":
        return load_mi_epochs_flexible(
            path, tmin=tmin, tmax=tmax, bandpass=bandpass,
            apply_car_ref=apply_car_ref, verbose=verbose,
        )
    raw = mne.io.read_raw_gdf(str(path), preload=False)
    _, event_id = mne.events_from_annotations(raw)
    has_four = all(str(c) in event_id for c in MI_EVENT_CODES)
    has_lr = all(str(c) in event_id for c in LR_EVENT_CODES)
    if has_four:
        return load_mi_epochs_flexible(
            path, tmin=tmin, tmax=tmax, bandpass=bandpass,
            apply_car_ref=apply_car_ref, verbose=verbose,
        )
    if has_lr:
        return load_mi_epochs_lr_flexible(
            path, tmin=tmin, tmax=tmax, bandpass=bandpass,
            apply_car_ref=apply_car_ref, verbose=verbose,
        )
    raise ValueError(f"无法识别 GDF 事件类型: {path}")


def load_unfiltered_epochs(
    gdf_path: Path | str,
    tmin: float = 0.5,
    tmax: float = 2.5,
) -> Tuple[np.ndarray, np.ndarray, float, List[str]]:
    """加载未滤波 epoch，供诊断统计使用。"""
    X, y, _, sfreq, ch_names = load_mi_epochs_flexible(
        gdf_path, tmin=tmin, tmax=tmax, bandpass=None, apply_car_ref=False, verbose=False
    )
    return X, y, sfreq, ch_names
