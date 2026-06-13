"""
MI 信号诊断：ERD/ERS 可视化 + CSP 特征 t-SNE

用于判断低准确率是否源于 BCI illiteracy（无可分离 MI 模式）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne.decoding import CSP
from scipy.signal import butter, filtfilt, hilbert
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from config import GDF_DIR, OUTPUTS_DIR, RANDOM_STATE
from src.gdf_loader import subject_to_gdf_path
from src.gdf_preprocessing import CLASS_NAMES, load_mi_epochs_flexible

# BCI 2a 通道名（GDF 中可能带 EEG- 前缀）
MOTOR_CHANNELS = ["C3", "Cz", "C4"]
CLASS_DISPLAY = {
    "foot": "Feet",
    "left": "Left Hand",
    "right": "Right Hand",
    "tongue": "Tongue",
}
# 各类别期望 ERD 最明显的通道（对侧或中线）
EXPECTED_ERD_CHANNEL = {
    "left": "C4",   # 左手 → 对侧 C4
    "right": "C3",  # 右手 → 对侧 C3
    "foot": "Cz",
    "tongue": "Cz",
}


def _resolve_channel_indices(ch_names: List[str], targets: List[str]) -> Dict[str, int]:
    """匹配 C3/Cz/C4 通道索引。"""
    indices = {}
    normalized = {name.upper().replace("EEG-", ""): i for i, name in enumerate(ch_names)}
    for ch in targets:
        key = ch.upper()
        if key in normalized:
            indices[ch] = normalized[key]
        else:
            for name, idx in normalized.items():
                if key in name:
                    indices[ch] = idx
                    break
    missing = set(targets) - set(indices)
    if missing:
        raise ValueError(f"未找到通道: {missing}，可用: {ch_names}")
    return indices


def _band_envelope_power(
    data: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """计算带限信号瞬时功率 (trials, ch, time)。"""
    nyq = sfreq / 2.0
    b, a = butter(4, [fmin / nyq, fmax / nyq], btype="band")
    filtered = filtfilt(b, a, data, axis=-1)
    analytic = hilbert(filtered, axis=-1)
    power = analytic.real ** 2 + analytic.imag ** 2
    # 50 ms 平滑
    win = max(int(0.05 * sfreq), 1)
    kernel = np.ones(win) / win
    smoothed = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="same"), -1, power)
    return smoothed


def compute_erds_timecourse(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    ch_names: List[str],
    fmin: float,
    fmax: float,
    baseline: Tuple[float, float] = (-0.5, 0.0),
    times: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    计算各类别 × 通道的 ERDS 时间曲线（% 相对基线）。

    返回:
        erds: (n_classes, n_channels, n_times) 百分比
        times: 时间轴
        ch_idx: 通道名 -> 索引
    """
    if times is None:
        times = np.arange(X.shape[2]) / sfreq + (-0.5 if X.shape[2] > 0 else 0)

    ch_idx = _resolve_channel_indices(ch_names, MOTOR_CHANNELS)
    ch_indices = [ch_idx[c] for c in MOTOR_CHANNELS]

    power = _band_envelope_power(X[:, ch_indices, :], sfreq, fmin, fmax)

    baseline_mask = (times >= baseline[0]) & (times < baseline[1])
    if not baseline_mask.any():
        raise ValueError("基线窗口为空")

    n_classes = len(CLASS_NAMES)
    erds = np.zeros((n_classes, len(MOTOR_CHANNELS), len(times)))

    for cls in range(n_classes):
        cls_power = power[y == cls]  # (trials, ch, time)
        if len(cls_power) == 0:
            continue
        mean_power = cls_power.mean(axis=0)  # (ch, time)
        baseline_power = mean_power[:, baseline_mask].mean(axis=1, keepdims=True)
        baseline_power = np.maximum(baseline_power, 1e-10)
        # dB 尺度，避免百分比在基线极小时爆炸
        erds[cls] = 10.0 * np.log10(mean_power / baseline_power)

    return erds, times, ch_idx


def compute_erds_summary_metrics(
    erds_mu: np.ndarray,
    erds_beta: np.ndarray,
    times: np.ndarray,
    active: Tuple[float, float] = (0.5, 2.5),
) -> pd.DataFrame:
    """
    提取 MI 窗口内期望通道的 ERD 强度（负值 = 能量下降）。
    """
    active_mask = (times >= active[0]) & (times <= active[1])
    rows = []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        expected_ch = EXPECTED_ERD_CHANNEL[cls_name]
        ch_idx = MOTOR_CHANNELS.index(expected_ch)
        mu_val = float(erds_mu[cls_idx, ch_idx, active_mask].mean())
        beta_val = float(erds_beta[cls_idx, ch_idx, active_mask].mean())
        rows.append({
            "class": cls_name,
            "expected_channel": expected_ch,
            "mu_erds_pct": mu_val,
            "beta_erds_pct": beta_val,
            "mu_erd_detected": mu_val < -1.0,   # dB 尺度：至少 1 dB 下降
            "beta_erd_detected": beta_val < -1.0,
        })
    return pd.DataFrame(rows)


def compute_contralateral_specificity(
    erds_mu: np.ndarray,
    times: np.ndarray,
    active: Tuple[float, float] = (0.5, 2.5),
) -> Dict[str, float]:
    """
    评估 ERD 是否具有类别特异性（对侧优势）。
    左手想象：C4 应比 C3 更负；右手想象：C3 应比 C4 更负。
    """
    active_mask = (times >= active[0]) & (times <= active[1])
    c3_i, _, c4_i = [MOTOR_CHANNELS.index(c) for c in MOTOR_CHANNELS]

    left_mu = erds_mu[CLASS_NAMES.index("left")][:, active_mask].mean(axis=1)
    right_mu = erds_mu[CLASS_NAMES.index("right")][:, active_mask].mean(axis=1)

    left_contrast = float(left_mu[c4_i] - left_mu[c3_i])
    right_contrast = float(right_mu[c3_i] - right_mu[c4_i])
    left_ok = left_contrast < -0.3
    right_ok = right_contrast < -0.3

    active_mu = erds_mu[:, :, active_mask].mean(axis=2)
    between_class_var = float(np.var(active_mu, axis=0).mean())

    return {
        "left_contrast_c4_minus_c3_db": left_contrast,
        "right_contrast_c3_minus_c4_db": right_contrast,
        "contralateral_pattern_correct": int(left_ok) + int(right_ok),
        "between_class_erds_variance": between_class_var,
    }


def plot_erds_figure(
    subject: str,
    erds_mu: np.ndarray,
    erds_beta: np.ndarray,
    times: np.ndarray,
    save_path: Path,
) -> None:
    """绘制 4 类 × 3 通道的 μ/β ERDS 图。"""
    fig, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=True)
    fig.suptitle(
        f"{subject} — ERD/ERS (dB change vs baseline -0.5~0s)\n"
        f"Negative = ERD (power decrease during MI)",
        fontsize=14,
        fontweight="bold",
    )

    for row, cls_idx in enumerate(range(4)):
        cls_name = CLASS_NAMES[cls_idx]
        display = CLASS_DISPLAY[cls_name]
        expected = EXPECTED_ERD_CHANNEL[cls_name]

        for col, (erds, band_label) in enumerate([(erds_mu, "μ (8-13 Hz)"), (erds_beta, "β (13-30 Hz)")]):
            ax = axes[row, col]
            for ch_i, ch_name in enumerate(MOTOR_CHANNELS):
                lw = 2.5 if ch_name == expected else 1.2
                alpha = 1.0 if ch_name == expected else 0.6
                ls = "-" if ch_name == expected else "--"
                ax.plot(times, erds[cls_idx, ch_i], label=ch_name, lw=lw, alpha=alpha, ls=ls)

            ax.axhline(0, color="gray", lw=0.8)
            ax.axvline(0, color="black", ls=":", lw=0.8, label="Cue")
            ax.axvspan(0.5, 2.5, alpha=0.08, color="green", label="MI window" if col == 0 and row == 0 else "")
            ax.set_ylabel(f"{display}\nERDS (dB)")
            if row == 0:
                ax.set_title(band_label)
            if col == 0:
                ax.annotate(
                    f"Expected ERD: {expected}",
                    xy=(0.02, 0.95), xycoords="axes fraction",
                    fontsize=9, color="darkred", va="top",
                )
            ax.legend(loc="lower right", fontsize=8)
            ax.set_ylim(-6, 4)
            ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ERDS 图已保存: {save_path}")


def extract_csp_features(X: np.ndarray, y: np.ndarray, n_components: int = 6) -> np.ndarray:
    """在全数据上 fit CSP（仅用于可视化诊断）。"""
    csp = CSP(n_components=n_components, reg="ledoit_wolf", log=True, norm_trace=False)
    csp.fit(X, y)
    return csp.transform(X)


def compute_tsne(features: np.ndarray, random_state: int = RANDOM_STATE) -> np.ndarray:
    """CSP 特征 → t-SNE 2D。"""
    X_scaled = StandardScaler().fit_transform(features)
    perplexity = min(30, max(5, len(features) // 10))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(X_scaled)


def plot_tsne_figure(
    subject: str,
    embeddings: np.ndarray,
    y: np.ndarray,
    silhouette: float,
    save_path: Path,
) -> None:
    """绘制 t-SNE 散点图。"""
    markers = ["o", "s", "^", "D"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, ax = plt.subplots(figsize=(8, 7))
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask = y == cls_idx
        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            c=colors[cls_idx],
            marker=markers[cls_idx],
            s=55,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.5,
            label=CLASS_DISPLAY[cls_name],
        )

    ax.set_title(
        f"{subject} — CSP features t-SNE\n"
        f"Silhouette score = {silhouette:.3f}  "
        f"({'clear clusters' if silhouette > 0.15 else 'mixed / no structure' if silhouette < 0.05 else 'weak structure'})",
        fontsize=12,
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  t-SNE 图已保存: {save_path}")


def analyze_subject(
    subject: str,
    gdf_dir: Path | None = None,
    out_dir: Path | None = None,
) -> Dict:
    """对单个受试者完成 ERDS + t-SNE 分析。"""
    out_dir = out_dir or OUTPUTS_DIR / "experiments" / "mi_signal_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    gdf_path = subject_to_gdf_path(subject, session="T", gdf_dir=gdf_dir or GDF_DIR)
    # 含基线 -0.5~0s，MI 至 4s，无滤波（ERDS 内部分带）
    X, y, _, sfreq, ch_names = load_mi_epochs_flexible(
        gdf_path, tmin=-0.5, tmax=4.0, bandpass=None, apply_car_ref=True
    )
    times = np.linspace(-0.5, 4.0, X.shape[2])

    print(f"\n--- {subject} ---")
    print(f"  trials={len(y)}, channels={X.shape[1]}, sfreq={sfreq} Hz")

    # ERDS
    erds_mu, times, _ = compute_erds_timecourse(X, y, sfreq, ch_names, 8, 13, times=times)
    erds_beta, _, _ = compute_erds_timecourse(X, y, sfreq, ch_names, 13, 30, times=times)
    plot_erds_figure(subject, erds_mu, erds_beta, times, out_dir / f"erds_{subject.lower()}.png")

    metrics_df = compute_erds_summary_metrics(erds_mu, erds_beta, times)
    metrics_df["subject"] = subject
    spec = compute_contralateral_specificity(erds_mu, times)
    n_erd_detected = int(metrics_df["mu_erd_detected"].sum() + metrics_df["beta_erd_detected"].sum())

    # t-SNE on bandpassed data for CSP
    X_bp, y_bp, _, _, _ = load_mi_epochs_flexible(
        gdf_path, tmin=0.5, tmax=2.5, bandpass=(8.0, 30.0), apply_car_ref=True
    )
    features = extract_csp_features(X_bp, y_bp, n_components=6)
    embeddings = compute_tsne(features)
    sil = float(silhouette_score(features, y_bp)) if len(np.unique(y_bp)) > 1 else 0.0
    plot_tsne_figure(subject, embeddings, y_bp, sil, out_dir / f"tsne_{subject.lower()}.png")

    return {
        "subject": subject,
        "n_trials": len(y),
        "silhouette_score": sil,
        "n_expected_erd_detected": n_erd_detected,
        "contralateral_pattern_correct": spec["contralateral_pattern_correct"],
        "left_contrast_db": spec["left_contrast_c4_minus_c3_db"],
        "right_contrast_db": spec["right_contrast_c3_minus_c4_db"],
        "between_class_erds_variance": spec["between_class_erds_variance"],
        "mu_erds_left_at_c4": metrics_df.loc[metrics_df["class"] == "left", "mu_erds_pct"].values[0],
        "mu_erds_right_at_c3": metrics_df.loc[metrics_df["class"] == "right", "mu_erds_pct"].values[0],
        "beta_erds_left_at_c4": metrics_df.loc[metrics_df["class"] == "left", "beta_erds_pct"].values[0],
        "beta_erds_right_at_c3": metrics_df.loc[metrics_df["class"] == "right", "beta_erds_pct"].values[0],
        "metrics_detail": metrics_df,
    }


def run_comparison(
    subjects: List[str] | None = None,
    gdf_dir: Path | None = None,
) -> pd.DataFrame:
    """对比多个受试者并生成汇总报告。"""
    subjects = subjects or ["A03", "A04", "A06"]
    out_dir = OUTPUTS_DIR / "experiments" / "mi_signal_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    all_metrics = []

    for subject in subjects:
        result = analyze_subject(subject, gdf_dir, out_dir)
        summary_rows.append({k: v for k, v in result.items() if k != "metrics_detail"})
        all_metrics.append(result["metrics_detail"])

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "mi_signal_summary.csv", index=False)
    pd.concat(all_metrics, ignore_index=True).to_csv(out_dir / "erds_metrics_detail.csv", index=False)

    # 文字报告
    lines = [
        "MI 信号可分离性诊断报告",
        "=" * 50,
        "",
        "【分析1: ERD/ERS — 类别特异性】",
        "关键不是「有没有 ERD」，而是「不同 MI 类别是否产生不同的对侧 ERD 模式」",
        "  左手想象 → C4 比 C3 更 ERD (contrast < -0.3 dB)",
        "  右手想象 → C3 比 C4 更 ERD (contrast < -0.3 dB)",
        "",
    ]
    for _, row in summary.iterrows():
        subj = row["subject"]
        lines.append(f"  {subj}:")
        lines.append(f"    对侧 ERD 模式正确数: {int(row['contralateral_pattern_correct'])}/2")
        lines.append(f"    左手 contrast (C4-C3): {row['left_contrast_db']:+.2f} dB")
        lines.append(f"    右手 contrast (C3-C4): {row['right_contrast_db']:+.2f} dB")
        lines.append(f"    跨类别 ERDS 方差: {row['between_class_erds_variance']:.3f}")
        lines.append("")

    lines += [
        "【分析2: CSP t-SNE Silhouette Score】",
        "  >0.15: 类别有明显簇结构",
        "  0.05~0.15: 弱结构",
        "  <0.05: 类别完全混杂（无可分离特征）",
        "",
    ]
    for _, row in summary.iterrows():
        sil = row["silhouette_score"]
        verdict = "明显分离" if sil > 0.15 else ("弱分离" if sil > 0.05 else "完全混杂")
        lines.append(f"  {row['subject']}: silhouette={sil:.3f} → {verdict}")

    lines += ["", "【结论】"]
    a03 = summary[summary["subject"] == "A03"]
    a04 = summary[summary["subject"] == "A04"]
    a06 = summary[summary["subject"] == "A06"]

    if not a03.empty and not a04.empty:
        a03_sil = a03.iloc[0]["silhouette_score"]
        a04_sil = a04.iloc[0]["silhouette_score"]
        a04_spec = int(a04.iloc[0]["contralateral_pattern_correct"])
        a03_spec = int(a03.iloc[0]["contralateral_pattern_correct"])
        if a03_sil > 0.15 and a04_sil < 0.05:
            lines.append("  A04 vs A03: CSP 特征无簇结构 → 分类器无法利用 MI 差异")
        if a04_spec < a03_spec:
            lines.append(f"  A04: 对侧 ERD 类别特异性 ({a04_spec}/2) 弱于 A03 ({a03_spec}/2)")
        if a04_sil < 0.05 and a04_spec <= 1:
            lines.append("  A04 综合判断: ERD 缺乏类别特异性 + CSP 特征混杂 → BCI illiteracy")

    if not a06.empty and not a03.empty:
        a06_sil = a06.iloc[0]["silhouette_score"]
        a06_spec = int(a06.iloc[0]["contralateral_pattern_correct"])
        if a06_sil < 0.05:
            lines.append("  A06: CSP 特征类别完全混杂")
        if a06_spec <= 1:
            lines.append(f"  A06: 对侧 ERD 类别特异性仅 {a06_spec}/2")
        if a06_sil < 0.05 and a06_spec <= 1:
            lines.append("  A06 综合判断: 符合 BCI illiteracy，换模型难以提升")

    report = "\n".join(lines)
    (out_dir / "mi_signal_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\n输出目录: {out_dir}")
    return summary
