"""
为每人最优配置生成混淆矩阵（5 折 CV 汇总 + 留出测试集）
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import OUTPUTS_DIR, RANDOM_STATE, TEST_SIZE
from src.gdf_trainer import (
    BAND_MAP,
    build_estimator,
    load_subject_data,
    plot_confusion_matrix,
)

EXP_DIR = OUTPUTS_DIR / "experiments"
CM_DIR = EXP_DIR / "confusion_matrices"


def save_cm_heatmap(cm: np.ndarray, class_names, title: str, path: Path) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  已保存: {path.name}")


def main() -> None:
    config_path = EXP_DIR / "best_config_per_subject.csv"
    if not config_path.exists():
        raise FileNotFoundError("请先运行 generate_experiment_report.py")

    best = pd.read_csv(config_path)
    CM_DIR.mkdir(parents=True, exist_ok=True)
    cv_rows = []

    print(f"为 {len(best)} 名受试者生成最优配置混淆矩阵...")
    for _, row in best.iterrows():
        subject = row["subject"]
        method = row["method"]
        band = row["band"]
        window = row["window"]
        use_car = bool(row["car"])

        X, y, class_names = load_subject_data(subject, band, window, use_car)
        pipeline = build_estimator(method, BAND_MAP.get(band, (8.0, 30.0)))

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cm_sum = np.zeros((len(class_names), len(class_names)), dtype=int)
        fold_accs = []
        for train_idx, test_idx in cv.split(X, y):
            est = build_estimator(method, BAND_MAP.get(band, (8.0, 30.0)))
            est.fit(X[train_idx], y[train_idx])
            pred = est.predict(X[test_idx])
            cm_sum += confusion_matrix(y[test_idx], pred, labels=np.arange(len(class_names)))
            fold_accs.append((pred == y[test_idx]).mean())

        cv_acc = float(np.mean(fold_accs))
        title = f"{subject} | {method} {band} {window} CAR={use_car} | CV={cv_acc:.1%}"
        save_cm_heatmap(cm_sum, class_names, title, CM_DIR / f"cm_best_{subject.lower()}_cv.png")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        test_acc = float((y_pred == y_test).mean())
        plot_confusion_matrix(
            y_test, y_pred, class_names,
            f"{subject} holdout acc={test_acc:.1%}",
            CM_DIR / f"cm_best_{subject.lower()}_holdout.png",
        )

        cv_rows.append({
            "subject": subject,
            "method": method,
            "band": band,
            "window": window,
            "car": use_car,
            "cv_accuracy": cv_acc,
            "holdout_accuracy": test_acc,
        })
        print(f"  {subject}: CV={cv_acc:.1%}, holdout={test_acc:.1%}")

    pd.DataFrame(cv_rows).to_csv(CM_DIR / "best_config_cm_summary.csv", index=False)
    print(f"\n混淆矩阵已保存至: {CM_DIR}")


if __name__ == "__main__":
    main()
