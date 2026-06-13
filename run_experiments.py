"""
系统化实验：诊断 + 频段/时间窗/CAR 网格 + CSP+SVM / FBCSP+LDA / EEGNet 对比
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import GDF_CSP_COMPONENTS, GDF_DIR, OUTPUTS_DIR, RANDOM_STATE
from src.experiment_eval import leakage_safe_cv_evaluate, verify_no_leakage_demo
from src.fbcsp import FBCSP_LDA
from src.gdf_diagnostics import print_diagnostics_summary, run_diagnostics
from src.gdf_loader import subject_to_gdf_path
from src.gdf_preprocessing import load_mi_epochs_flexible

try:
    from src.eegnet import EEGNetClassifier
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

BAND_OPTIONS = {
    "8-30Hz": (8.0, 30.0),
    "7-35Hz": (7.0, 35.0),
    "4-40Hz": (4.0, 40.0),
}

WINDOW_OPTIONS = {
    "0.5-2.5s": (0.5, 2.5),
    "0.5-3.5s": (0.5, 3.5),
    "0.5-4.0s": (0.5, 4.0),
    "1.0-4.0s": (1.0, 4.0),
}

METHODS = ["CSP+SVM", "FBCSP+LDA", "EEGNet"]


def build_estimator(method: str, bandpass: tuple, sample_rate: float = 250.0):
    """创建指定方法的新 estimator 实例（供 CV 每折 clone）。"""
    if method == "CSP+SVM":
        return Pipeline([
            ("csp", CSP(GDF_CSP_COMPONENTS, reg="ledoit_wolf", log=True)),
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", random_state=RANDOM_STATE)),
        ])
    if method == "FBCSP+LDA":
        return FBCSP_LDA(bandpass=bandpass, n_components=2, sample_rate=sample_rate)
    if method == "EEGNet":
        if not HAS_TORCH:
            raise ImportError("EEGNet 需要 PyTorch，请运行: pip install torch")
        return EEGNetClassifier(n_epochs_train=60, batch_size=32, random_state=RANDOM_STATE)
    raise ValueError(method)


def plot_cm(cm, class_names, title, path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def run_single_experiment(
    subject: str,
    band_name: str,
    bandpass: tuple,
    window_name: str,
    tmin: float,
    tmax: float,
    use_car: bool,
    method: str,
    out_dir: Path,
    save_cm: bool = False,
) -> dict:
    """运行单个受试者+配置的泄漏安全 CV。"""
    path = subject_to_gdf_path(subject, session="T", gdf_dir=GDF_DIR)
    X, y, class_names, sfreq, _ = load_mi_epochs_flexible(
        path, tmin=tmin, tmax=tmax, bandpass=bandpass, apply_car_ref=use_car, verbose=False
    )

    result = leakage_safe_cv_evaluate(
        X, y,
        lambda: build_estimator(method, bandpass, sfreq),
        random_state=RANDOM_STATE,
        class_names=class_names,
    )

    row = {
        "subject": subject,
        "method": method,
        "band": band_name,
        "window": window_name,
        "car": use_car,
        "accuracy": result["accuracy"],
        "kappa": result["kappa"],
        "fold_accuracy_std": result["fold_accuracy_std"],
    }

    if save_cm:
        cm_dir = out_dir / "confusion_matrices"
        cm_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{subject}_{method}_{band_name}_{window_name}_CAR{use_car}".replace("+", "_")
        plot_cm(result["confusion_matrix"], class_names, tag, cm_dir / f"{tag}.png")

    return row


def run_grid(
    subjects: list,
    methods: list,
    include_eegnet: bool = True,
    quick: bool = False,
) -> pd.DataFrame:
    """运行完整实验网格。"""
    out_dir = OUTPUTS_DIR / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    total = len(subjects) * len(BAND_OPTIONS) * len(WINDOW_OPTIONS) * 2 * len(methods)
    done = 0

    for subject in subjects:
        for band_name, bandpass in BAND_OPTIONS.items():
            for window_name, (tmin, tmax) in WINDOW_OPTIONS.items():
                for use_car in [False, True]:
                    for method in methods:
                        if method == "EEGNet" and not include_eegnet:
                            continue
                        if quick and method == "EEGNet":
                            continue
                        t0 = time.time()
                        try:
                            save_cm = (band_name == "8-30Hz" and window_name == "0.5-2.5s" and not use_car)
                            row = run_single_experiment(
                                subject, band_name, bandpass, window_name,
                                tmin, tmax, use_car, method, out_dir, save_cm=save_cm,
                            )
                            row["elapsed_sec"] = round(time.time() - t0, 1)
                            rows.append(row)
                            print(
                                f"[{done+1}] {subject} {method} {band_name} {window_name} "
                                f"CAR={use_car} -> acc={row['accuracy']:.3f} kappa={row['kappa']:.3f}"
                            )
                        except Exception as exc:
                            print(f"FAIL {subject} {method} {band_name} {window_name} CAR={use_car}: {exc}")
                        done += 1

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "all_results.csv", index=False)
    return df


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """生成汇总表：各方法平均、最佳配置/受试者。"""
    out_dir = OUTPUTS_DIR / "experiments"

    method_mean = df.groupby("method")[["accuracy", "kappa"]].mean().round(4)
    method_mean.to_csv(out_dir / "method_mean_summary.csv")

    best_per_subject = (
        df.sort_values("accuracy", ascending=False)
        .groupby("subject", as_index=False)
        .first()
    )
    best_per_subject.to_csv(out_dir / "best_config_per_subject.csv", index=False)

    pivot = df.pivot_table(
        index="subject", columns="method", values="accuracy", aggfunc="max"
    ).round(4)
    pivot["mean"] = pivot.mean(axis=1).round(4)
    pivot.to_csv(out_dir / "subject_max_accuracy_by_method.csv")

    overall_mean = df.groupby("method")["accuracy"].mean()
    print("\n========== 方法平均 Accuracy ==========")
    print(overall_mean.round(4).to_string())
    print(f"\n全局最高单实验 Accuracy: {df['accuracy'].max():.4f}")
    print(f"按受试者最优配置的平均 Accuracy: {best_per_subject['accuracy'].mean():.4f}")
    print(f"当前默认配置(8-30,0.5-2.5,noCAR) CSP+SVM 平均: ", end="")
    sub = df[
        (df["band"] == "8-30Hz") & (df["window"] == "0.5-2.5s")
        & (df["car"] == False) & (df["method"] == "CSP+SVM")
    ]
    if not sub.empty:
        print(f"{sub['accuracy'].mean():.4f}")

    return best_per_subject


def run_leakage_audit(subject: str = "A01") -> None:
    path = subject_to_gdf_path(subject)
    X, y, _, _, _ = load_mi_epochs_flexible(path, bandpass=(8, 30))
    report = verify_no_leakage_demo(X, y, random_state=RANDOM_STATE)
    out = OUTPUTS_DIR / "experiments" / "leakage_audit.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)


def main():
    parser = argparse.ArgumentParser(description="BCI 2a 系统化实验")
    parser.add_argument("--subjects", type=str, default="all", help="如 all 或 1,3,7")
    parser.add_argument("--quick", action="store_true", help="跳过 EEGNet 加速")
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument("--skip-grid", action="store_true")
    args = parser.parse_args()

    if args.subjects == "all":
        subjects = [f"A{i:02d}" for i in range(1, 10)]
    else:
        subjects = [f"A{int(s.strip()):02d}" for s in args.subjects.split(",")]

    print("=" * 60)
    print("  BCI 2a 系统化实验")
    print("=" * 60)

    # 1. 泄漏检查
    run_leakage_audit("A01")

    # 2. 受试者诊断
    diag = run_diagnostics(GDF_DIR)
    print_diagnostics_summary(diag)

    if args.diagnostics_only:
        return

    if args.skip_grid:
        return

    # 3. 实验网格
    methods = ["CSP+SVM", "FBCSP+LDA"]
    if not args.quick:
        methods.append("EEGNet")

    print(f"\n开始实验网格: {len(subjects)} 受试者, 方法={methods}")
    df = run_grid(subjects, methods, include_eegnet=not args.quick, quick=args.quick)
    summarize_results(df)
    print(f"\n结果已保存至: {OUTPUTS_DIR / 'experiments'}")


if __name__ == "__main__":
    main()
