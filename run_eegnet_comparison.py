"""
EEGNet vs CSP/FBCSP 对比实验（5 折泄漏安全 CV）

用法:
    python run_eegnet_comparison.py
    python run_eegnet_comparison.py --subjects A01,A03
    python run_eegnet_comparison.py --epochs 60
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import GDF_DIR, OUTPUTS_DIR, RANDOM_STATE
from src.eegnet import EEGNetClassifier
from src.experiment_eval import leakage_safe_cv_evaluate
from src.gdf_loader import subject_to_gdf_path
from src.gdf_preprocessing import load_mi_epochs_flexible
from src.gdf_trainer import BAND_MAP, WINDOW_MAP

OUT_DIR = OUTPUTS_DIR / "experiments" / "eegnet_comparison"
BASELINE_CSV = OUTPUTS_DIR / "experiments" / "best_config_per_subject.csv"
CLASS_NAMES = ["foot", "left", "right", "tongue"]


def load_epochs(subject: str, band: str, window: str, use_car: bool):
    path = subject_to_gdf_path(subject, session="T", gdf_dir=GDF_DIR)
    bandpass = BAND_MAP.get(band, (8.0, 30.0))
    tmin, tmax = WINDOW_MAP.get(window, (0.5, 2.5))
    X, y, _, sfreq, _ = load_mi_epochs_flexible(
        path, tmin=tmin, tmax=tmax, bandpass=bandpass, apply_car_ref=use_car
    )
    return X, y, sfreq


def run_eegnet_cv(
    subject: str,
    band: str,
    window: str,
    use_car: bool,
    n_epochs_train: int,
    config_label: str,
) -> dict:
    X, y, sfreq = load_epochs(subject, band, window, use_car)
    t0 = time.time()

    def factory():
        return EEGNetClassifier(
            n_epochs_train=n_epochs_train,
            batch_size=32,
            learning_rate=1e-3,
            random_state=RANDOM_STATE,
        )

    result = leakage_safe_cv_evaluate(X, y, factory, random_state=RANDOM_STATE, class_names=CLASS_NAMES)
    elapsed = round(time.time() - t0, 1)

    return {
        "subject": subject,
        "method": "EEGNet",
        "config_label": config_label,
        "band": band,
        "window": window,
        "car": use_car,
        "accuracy": result["accuracy"],
        "kappa": result["kappa"],
        "fold_accuracy_std": result["fold_accuracy_std"],
        "fold_accuracies": result["fold_accuracies"],
        "elapsed_sec": elapsed,
        "confusion_matrix": result["confusion_matrix"],
    }


def plot_comparison(summary: pd.DataFrame, out_path: Path) -> None:
    plot_df = summary.melt(
        id_vars=["subject"],
        value_vars=["baseline_accuracy", "eegnet_default_acc", "eegnet_tuned_acc"],
        var_name="method",
        value_name="accuracy",
    )
    label_map = {
        "baseline_accuracy": "Best CSP/FBCSP (grid)",
        "eegnet_default_acc": "EEGNet (8-30Hz, 0.5-2.5s)",
        "eegnet_tuned_acc": "EEGNet (subject band/window)",
    }
    plot_df["method"] = plot_df["method"].map(label_map)

    plt.figure(figsize=(12, 5))
    sns.barplot(data=plot_df, x="subject", y="accuracy", hue="method")
    plt.axhline(0.25, color="gray", linestyle="--", linewidth=0.8, label="chance (4-class)")
    plt.ylim(0, 1.0)
    plt.ylabel("5-fold CV Accuracy")
    plt.title("EEGNet vs Best Traditional Pipeline (BCI 2a)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_cm(cm, subject: str, title: str, path: Path) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="EEGNet 对比实验")
    parser.add_argument("--subjects", type=str, default="all")
    parser.add_argument("--epochs", type=int, default=80, help="EEGNet 每折训练 epoch 数")
    args = parser.parse_args()

    if args.subjects == "all":
        subjects = [f"A{i:02d}" for i in range(1, 10)]
    else:
        subjects = [s.strip().upper() for s in args.subjects.split(",")]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_csv(BASELINE_CSV) if BASELINE_CSV.exists() else None

    all_rows = []
    summary_rows = []

    print("=" * 60)
    print("  EEGNet 对比实验 (5-fold leakage-safe CV)")
    print(f"  subjects={subjects}  epochs={args.epochs}")
    print("=" * 60)

    for subject in subjects:
        print(f"\n----- {subject} -----")

        # 1) 默认配置（与 CSP+SVM 默认一致）
        r_default = run_eegnet_cv(
            subject, "8-30Hz", "0.5-2.5s", False, args.epochs, "default"
        )
        all_rows.append({k: v for k, v in r_default.items() if k != "confusion_matrix"})
        print(
            f"  EEGNet default: acc={r_default['accuracy']:.1%} "
            f"kappa={r_default['kappa']:.3f} ({r_default['elapsed_sec']}s)"
        )
        plot_cm(
            r_default["confusion_matrix"],
            subject,
            f"EEGNet default - {subject}",
            OUT_DIR / f"cm_{subject}_eegnet_default.png",
        )

        # 2) 使用该受试者 grid 搜索最优的 band/window/car（方法仍用 EEGNet）
        r_tuned = r_default
        if baseline is not None and subject in baseline["subject"].values:
            row = baseline[baseline["subject"] == subject].iloc[0]
            r_tuned = run_eegnet_cv(
                subject,
                row["band"],
                row["window"],
                bool(row["car"]),
                args.epochs,
                "subject_preprocess",
            )
            all_rows.append({k: v for k, v in r_tuned.items() if k != "confusion_matrix"})
            print(
                f"  EEGNet tuned ({row['band']} {row['window']} CAR={row['car']}): "
                f"acc={r_tuned['accuracy']:.1%} kappa={r_tuned['kappa']:.3f} "
                f"({r_tuned['elapsed_sec']}s)"
            )
            plot_cm(
                r_tuned["confusion_matrix"],
                subject,
                f"EEGNet tuned - {subject}",
                OUT_DIR / f"cm_{subject}_eegnet_tuned.png",
            )

        best_eegnet = r_tuned if r_tuned["accuracy"] >= r_default["accuracy"] else r_default
        baseline_acc = float(baseline[baseline["subject"] == subject]["accuracy"].iloc[0]) if baseline is not None else None
        baseline_method = baseline[baseline["subject"] == subject]["method"].iloc[0] if baseline is not None else "?"

        summary_rows.append(
            {
                "subject": subject,
                "baseline_method": baseline_method,
                "baseline_accuracy": baseline_acc,
                "eegnet_default_acc": r_default["accuracy"],
                "eegnet_tuned_acc": r_tuned["accuracy"],
                "eegnet_best_acc": best_eegnet["accuracy"],
                "eegnet_best_config": f"{best_eegnet['band']} {best_eegnet['window']} CAR={best_eegnet['car']}",
                "eegnet_vs_baseline_pp": (
                    round((best_eegnet["accuracy"] - baseline_acc) * 100, 2) if baseline_acc else None
                ),
            }
        )

    results_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(summary_rows)

    results_df.to_csv(OUT_DIR / "eegnet_all_runs.csv", index=False)
    summary_df.to_csv(OUT_DIR / "eegnet_vs_baseline_summary.csv", index=False)
    plot_comparison(summary_df, OUT_DIR / "eegnet_comparison_barplot.png")

    report_lines = [
        "EEGNet vs CSP/FBCSP 对比报告",
        "=" * 50,
        f"EEGNet 训练 epoch/折: {args.epochs}",
        f"评估: 5-fold stratified CV（无泄漏）",
        "",
        "各受试者对比:",
    ]
    for _, r in summary_df.iterrows():
        delta = r["eegnet_vs_baseline_pp"]
        sign = f"{delta:+.1f}pp" if delta is not None else "N/A"
        report_lines.append(
            f"  {r['subject']}: baseline {r['baseline_method']} {r['baseline_accuracy']:.1%} | "
            f"EEGNet best {r['eegnet_best_acc']:.1%} ({sign})"
        )

    report_lines.extend(
        [
            "",
            f"EEGNet 默认配置平均: {summary_df['eegnet_default_acc'].mean():.1%}",
            f"EEGNet 调参预处理平均: {summary_df['eegnet_tuned_acc'].mean():.1%}",
            f"EEGNet 每人最优平均: {summary_df['eegnet_best_acc'].mean():.1%}",
            f"CSP/FBCSP grid 最优平均: {summary_df['baseline_accuracy'].mean():.1%}",
            "",
            f"EEGNet 胜 baseline 人数: {(summary_df['eegnet_vs_baseline_pp'] > 0).sum()}/{len(summary_df)}",
        ]
    )
    report_text = "\n".join(report_lines)
    (OUT_DIR / "eegnet_comparison_report.txt").write_text(report_text, encoding="utf-8")

    print("\n" + report_text)
    print(f"\n结果目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
